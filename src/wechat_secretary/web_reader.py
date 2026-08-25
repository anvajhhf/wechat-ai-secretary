from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
from dataclasses import dataclass
from enum import StrEnum
from html.parser import HTMLParser
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from .config import SecretarySettings
from .models import IntentKind


_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = "。！？；，、.!?;,)]}）】》」』"
_DEEP_LINK_CUE = re.compile(
    r"(?:深度笔记|深度(?:分析|整理|研究)|深入(?:分析|整理|研究)|"
    r"详细(?:分析|整理|研究)|做(?:一份|个)?深度笔记)",
    re.IGNORECASE,
)
_NORMAL_LINK_CUE = re.compile(
    r"(?:记一下|记下来|保存|收藏|做(?:一份|个)?笔记|整理一下|总结一下|"
    r"分析一下|看一下|看看|帮我(?:整理|总结|记录|看看))",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_KEY = re.compile(
    r"(?:token|secret|auth|sign(?:ature)?|session|credential|password|passwd|"
    r"ticket|cookie|jwt|oauth|skey|access[_-]?key|api[_-]?key)",
    re.IGNORECASE,
)
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "spm",
    "yclid",
}


class LinkNoteMode(StrEnum):
    NONE = "none"
    ASK = "ask"
    NORMAL = "normal"
    DEEP = "deep"


@dataclass(frozen=True)
class LinkNoteDecision:
    mode: LinkNoteMode
    urls: tuple[str, ...] = ()


def extract_web_urls(text: str) -> tuple[str, ...]:
    urls: list[str] = []
    for match in _URL_PATTERN.finditer(text or ""):
        value = match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
        if value and value not in urls:
            urls.append(value)
    return tuple(urls)


def sanitize_web_url(url: str) -> str:
    """Remove credentials, share tokens, trackers and fragments from display URLs."""

    candidate = (url or "").strip()
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return ""
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return ""
    try:
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return ""
    if not host:
        return ""
    host_for_url = f"[{host}]" if ":" in host else host
    netloc = host_for_url if port is None else f"{host_for_url}:{port}"
    kept: list[tuple[str, str]] = []
    try:
        pairs = parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=False,
            max_num_fields=100,
        )
    except ValueError:
        pairs = []
    for key, value in pairs:
        lowered = key.casefold().strip()
        tracking = lowered.startswith(("utm_", "xsec_")) or lowered in _TRACKING_QUERY_KEYS
        sensitive = bool(_SENSITIVE_QUERY_KEY.search(lowered))
        nested_sensitive = bool(
            re.search(
                r"(?:token|secret|auth|signature|session|password|credential)=",
                value,
                re.IGNORECASE,
            )
        )
        if tracking or sensitive or nested_sensitive:
            continue
        kept.append((key, value))
    query = urlencode(kept, doseq=True)
    return urlunsplit((parsed.scheme.casefold(), netloc, parsed.path or "/", query, ""))


def sanitize_web_urls_in_text(text: str) -> str:
    """Sanitize every HTTP(S) URL while preserving surrounding punctuation."""

    def replace_url(match: re.Match[str]) -> str:
        matched = match.group(0)
        raw = matched.rstrip(_TRAILING_URL_PUNCTUATION)
        suffix = matched[len(raw) :]
        sanitized = sanitize_web_url(raw)
        return (sanitized or "[链接参数已隐藏]") + suffix

    return _URL_PATTERN.sub(replace_url, text or "")


def decide_link_note(
    text: str,
    forced_kind: IntentKind | None,
    deep_note: bool,
) -> LinkNoteDecision:
    urls = extract_web_urls(text)
    if not urls or forced_kind is IntentKind.TASK:
        return LinkNoteDecision(LinkNoteMode.NONE, urls)
    if deep_note:
        return LinkNoteDecision(LinkNoteMode.DEEP, urls)
    if forced_kind is IntentKind.NOTE:
        return LinkNoteDecision(LinkNoteMode.NORMAL, urls)
    if _DEEP_LINK_CUE.search(text):
        return LinkNoteDecision(LinkNoteMode.DEEP, urls)
    if _NORMAL_LINK_CUE.search(text):
        return LinkNoteDecision(LinkNoteMode.NORMAL, urls)
    remaining = _URL_PATTERN.sub("", text).strip(" \t\r\n，。！？；、:：,.!?;()（）[]【】")
    if not remaining:
        return LinkNoteDecision(LinkNoteMode.ASK, urls)
    return LinkNoteDecision(LinkNoteMode.NONE, urls)


class WebReadError(ValueError):
    pass


@dataclass(frozen=True)
class WebPage:
    source_url: str
    final_url: str
    title: str
    text: str


def sanitize_web_page(page: WebPage) -> WebPage:
    """Enforce the model/note boundary even for alternate reader implementations."""

    return WebPage(
        source_url=sanitize_web_url(page.source_url),
        final_url=sanitize_web_url(page.final_url),
        title=sanitize_web_urls_in_text(page.title),
        text=sanitize_web_urls_in_text(page.text),
    )


@dataclass(frozen=True)
class FetchResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class WebReader(Protocol):
    def read(self, url: str) -> WebPage: ...


class DisabledWebReader:
    def read(self, url: str) -> WebPage:
        del url
        raise WebReadError("网页读取功能尚未启用")


class _HTMLTextExtractor(HTMLParser):
    _IGNORED = {"script", "style", "noscript", "template", "svg", "canvas"}
    _BLOCKS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered in self._IGNORED:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if lowered == "title":
            self._in_title = True
        if lowered in self._BLOCKS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in self._IGNORED:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if lowered == "title":
            self._in_title = False
        if lowered in self._BLOCKS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._parts.append(data)

    @staticmethod
    def _clean(parts: Sequence[str], limit: int) -> str:
        lines: list[str] = []
        for raw in "".join(parts).splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if line and (not lines or lines[-1] != line):
                lines.append(line)
        value = "\n".join(lines).strip()
        if len(value) <= limit:
            return value
        clipped = value[:limit]
        boundary = max(clipped.rfind("\n"), clipped.rfind("。"), clipped.rfind(" "))
        if boundary >= int(limit * 0.75):
            clipped = clipped[:boundary]
        return clipped.rstrip() + "\n[网页正文已按安全长度截断]"

    def result(self, text_limit: int) -> tuple[str, str]:
        title = self._clean(self._title_parts, 200)
        return title, self._clean(self._parts, text_limit)


Resolver = Callable[[str, int], Sequence[str]]
Requester = Callable[[str, tuple[str, ...], int, float], FetchResponse]


class SafeWebReader:
    """Read one public HTML/text URL without cookies, proxies, scripts or private IPs."""

    def __init__(
        self,
        settings: SecretarySettings,
        *,
        resolver: Resolver | None = None,
        requester: Requester | None = None,
    ) -> None:
        self.settings = settings
        self._resolver = resolver or self._resolve
        self._requester = requester or self._request

    @staticmethod
    def _resolve(host: str, port: int) -> tuple[str, ...]:
        try:
            rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise WebReadError("域名解析失败") from exc
        addresses: list[str] = []
        for row in rows:
            address = str(row[4][0]).split("%", 1)[0]
            if address not in addresses:
                addresses.append(address)
        return tuple(addresses)

    def _public_addresses(self, addresses: Sequence[str]) -> tuple[str, ...]:
        if not addresses:
            raise WebReadError("域名没有可用地址")
        checked: list[str] = []
        for raw in addresses:
            try:
                address = ipaddress.ip_address(raw)
            except ValueError as exc:
                raise WebReadError("域名返回了无效地址") from exc
            proxy_fake_ip = address in ipaddress.ip_network("198.18.0.0/15")
            if not address.is_global and not (
                self.settings.web_allow_proxy_fake_ip and proxy_fake_ip
            ):
                raise WebReadError("链接指向本机、内网或保留地址，已拒绝读取")
            checked.append(str(address))
        return tuple(checked)

    def _validated_target(self, url: str) -> tuple[str, tuple[str, ...]]:
        candidate = (url or "").strip()
        if not candidate or len(candidate) > 2048 or any(ord(char) < 32 for char in candidate):
            raise WebReadError("链接格式无效或过长")
        try:
            parsed = urlsplit(candidate)
            port = parsed.port
        except ValueError as exc:
            raise WebReadError("链接端口格式无效") from exc
        scheme = parsed.scheme.casefold()
        if scheme not in {"http", "https"}:
            raise WebReadError("只支持公开的 HTTP 或 HTTPS 网页")
        if parsed.username is not None or parsed.password is not None:
            raise WebReadError("链接不得包含账号或密码")
        host = (parsed.hostname or "").rstrip(".")
        if not host:
            raise WebReadError("链接缺少有效域名")
        try:
            host = host.encode("idna").decode("ascii").casefold()
        except UnicodeError as exc:
            raise WebReadError("链接域名格式无效") from exc
        expected_port = 443 if scheme == "https" else 80
        port = port or expected_port
        if port != expected_port:
            raise WebReadError("为安全起见，只允许标准网页端口")
        addresses = self._public_addresses(self._resolver(host, port))
        host_for_url = f"[{host}]" if ":" in host else host
        path = parsed.path or "/"
        normalized = urlunsplit((scheme, host_for_url, path, parsed.query, ""))
        return normalized, addresses

    @staticmethod
    def _request(
        url: str,
        addresses: tuple[str, ...],
        max_bytes: int,
        timeout: float,
    ) -> FetchResponse:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"
        last_error: BaseException | None = None
        for address in addresses:
            connection: http.client.HTTPConnection | None = None
            try:
                if parsed.scheme == "https":
                    connection = http.client.HTTPSConnection(
                        host,
                        port,
                        timeout=timeout,
                        context=ssl.create_default_context(),
                    )
                else:
                    connection = http.client.HTTPConnection(host, port, timeout=timeout)

                def pinned_connection(
                    ignored_address: tuple[str, int],
                    connect_timeout: float | object = timeout,
                    source_address: tuple[str, int] | None = None,
                ) -> socket.socket:
                    del ignored_address
                    actual_timeout = timeout if not isinstance(connect_timeout, (int, float)) else connect_timeout
                    return socket.create_connection(
                        (address, port), actual_timeout, source_address
                    )

                connection._create_connection = pinned_connection  # type: ignore[attr-defined]
                connection.request(
                    "GET",
                    target,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                        "Accept-Encoding": "identity",
                        "Connection": "close",
                        "User-Agent": "WechatAISecretary/0.1 (personal link note reader)",
                    },
                )
                response = connection.getresponse()
                raw_length = response.getheader("Content-Length", "").strip()
                if raw_length.isdigit() and int(raw_length) > max_bytes:
                    raise WebReadError("网页内容超过安全大小限制")
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise WebReadError("网页内容超过安全大小限制")
                headers = {key.casefold(): value for key, value in response.getheaders()}
                return FetchResponse(int(response.status), headers, body)
            except WebReadError:
                raise
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                if connection is not None:
                    connection.close()
        raise WebReadError("网页连接失败") from last_error

    @staticmethod
    def _decode(body: bytes, content_type: str) -> str:
        match = re.search(r"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", content_type, re.I)
        if match is None:
            head = body[:4096].decode("ascii", errors="ignore")
            match = re.search(r"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", head, re.I)
        encodings = [match.group(1)] if match is not None else []
        encodings.extend(["utf-8", "gb18030"])
        for encoding in dict.fromkeys(encodings):
            try:
                return body.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return body.decode("utf-8", errors="replace")

    def read(self, url: str) -> WebPage:
        if not self.settings.web_enabled:
            raise WebReadError("网页读取功能尚未启用")
        current = url
        source_url = url
        visited: set[str] = set()
        for redirect_index in range(self.settings.web_max_redirects + 1):
            current, addresses = self._validated_target(current)
            if current in visited:
                raise WebReadError("网页发生循环跳转")
            visited.add(current)
            response = self._requester(
                current,
                addresses,
                self.settings.web_max_bytes,
                float(self.settings.web_timeout_seconds),
            )
            if response.status in {301, 302, 303, 307, 308}:
                location = str(response.headers.get("location", "")).strip()
                if not location:
                    raise WebReadError("网页跳转缺少目标地址")
                if redirect_index >= self.settings.web_max_redirects:
                    raise WebReadError("网页跳转次数过多")
                current = urljoin(current, location)
                continue
            if response.status < 200 or response.status >= 300:
                raise WebReadError(f"网页返回 HTTP {response.status}")
            encoding = str(response.headers.get("content-encoding", "identity")).casefold()
            if encoding not in {"", "identity"}:
                raise WebReadError("网页返回了不支持的压缩格式")
            content_type = str(response.headers.get("content-type", "")).casefold()
            media_type = content_type.split(";", 1)[0].strip()
            allowed = {"text/html", "application/xhtml+xml", "text/plain"}
            if media_type not in allowed:
                raise WebReadError("当前只支持公开的 HTML 或纯文本网页")
            decoded = self._decode(response.body, content_type)
            if media_type == "text/plain":
                title = urlsplit(current).hostname or "网页笔记"
                text = _HTMLTextExtractor._clean((decoded,), self.settings.web_max_text_chars)
            else:
                parser = _HTMLTextExtractor()
                parser.feed(decoded)
                parser.close()
                title, text = parser.result(self.settings.web_max_text_chars)
                title = title or (urlsplit(current).hostname or "网页笔记")
            if len(text.strip()) < 20:
                raise WebReadError("网页没有足够的可读正文")
            return sanitize_web_page(
                WebPage(
                    source_url=source_url,
                    final_url=current,
                    title=title,
                    text=text,
                )
            )
        raise WebReadError("网页跳转次数过多")
