"""Only synthetic Markdown, indexes and controls are used in these tests."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from wechat_secretary.notes_search import (
    NotesIndex, NotesQuery, NotesSnapshot, NotesUnavailable, has_notes_state,
)


TEST_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "test-temp"
NOW = datetime(2035, 1, 15, tzinfo=timezone.utc)


def note(title, body, *, ref="1234567890abcdef", recorded="2035-01-12T14:00:00+08:00", tags=""):
    return f"## {title}\n\n{body}\n\n标签：{tags}\n\n> 来源：微信｜时间：{recorded}｜引用：{ref}\n<!-- wechat-ai-secretary:ref={ref} -->\n"


class NotesFixture(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = (TEST_ROOT / f"notes-search-{uuid4().hex}").resolve()
        self.root.mkdir()
        self.vault = self.root / "Vault"
        self.vault.mkdir()
        self.index_path = self.root / "state" / "notes-index.sqlite3"
        self.control_path = self.root / "control" / "control.json"
        self.addCleanup(self.cleanup_fixture)

    def cleanup_fixture(self):
        if self.root.parent != TEST_ROOT.resolve() or not self.root.name.startswith("notes-search-"):
            raise AssertionError("refusing to clean outside owned fixture")
        shutil.rmtree(self.root)

    def write(self, relative, text):
        path = self.vault / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def store(self, **kwargs):
        options = dict(vault_path=self.vault, allowed_paths=("Inbox",), control_path=self.control_path, clock=lambda: NOW)
        options.update(kwargs)
        index = NotesIndex(self.index_path, **options)
        self.addCleanup(index.close)
        return index

    def hits(self, store, text="", **kwargs):
        return store.search(NotesQuery(text, **kwargs)).hits


class NotesRetrievalTests(NotesFixture):
    def test_two_character_chinese_and_mixed_english_find_expected_note(self):
        self.write("Inbox/a.md", note("免疫实验", "用 CRISPR Cas9 敲除基因后观察免疫反应"))
        self.write("Inbox/b.md", note("购物", "水果和牛奶", ref="abcdef1234567890"))
        index = self.store()
        for query in ("免疫", "CRISPR", "免疫 Cas9", "免"):
            with self.subTest(query=query):
                self.assertEqual("免疫实验", self.hits(index, query)[0].title)
        self.assertEqual((), self.hits(index, "完全不存在的词汇xyz"))

    def test_title_ranks_before_body_and_limit_is_five(self):
        for number in range(8):
            self.write(f"Inbox/{number}.md", note("实验方案" if number == 7 else f"记录{number}", "实验方案详见这里", ref=f"{number:016x}"))
        hits = self.hits(self.store(), "实验方案")
        self.assertEqual(5, len(hits))
        self.assertEqual("实验方案", hits[0].title)

    def test_shared_inbox_separates_notes_and_keeps_source_locations(self):
        first = note("相同标题", "第一份实验方案", ref="1111111111111111", recorded="2035-01-01T10:00:00+08:00", tags="#生物 #实验")
        second = note("相同标题", "第二份实验方案", ref="2222222222222222", recorded="2035-01-02T11:00:00+08:00", tags="#化学")
        self.write("Inbox/微信收件箱.md", first + "\n" + second)
        hits = self.hits(self.store(), "实验方案")
        self.assertEqual(2, len(hits))
        self.assertEqual(1, len({hit.source_id for hit in hits}))
        self.assertEqual(2, len({hit.chunk_id for hit in hits}))
        self.assertEqual({"1111111111111111", "2222222222222222"}, {hit.source_ref for hit in hits})
        self.assertTrue(all(hit.line_start <= hit.line_end for hit in hits))
        second_hit = next(hit for hit in hits if hit.source_ref == "2222222222222222")
        self.assertIn("第二份", second_hit.excerpt)
        self.assertNotIn("第一份", second_hit.excerpt)
        self.assertGreater(second_hit.line_start, 1)

    def test_tags_and_inclusive_record_dates_filter(self):
        self.write("Inbox/a.md", note("实验", "内容", tags="#生物 #RNA", recorded="2035-01-12T10:00:00+08:00"))
        index = self.store()
        self.assertEqual(1, len(self.hits(index, tags=("生物", "rna"), date_from=date(2035, 1, 12), date_to=date(2035, 1, 12))))
        self.assertEqual((), self.hits(index, tags=("化学",)))
        self.assertEqual((), self.hits(index, date_to=date(2035, 1, 11)))

    def test_unknown_date_does_not_use_mtime(self):
        path = self.write("Inbox/undated.md", "## 实验\n未知日期的实验记录")
        os.utime(path, (NOW.timestamp(), NOW.timestamp()))
        index = self.store()
        self.assertEqual("", self.hits(index, "实验")[0].recorded_at)
        self.assertEqual((), self.hits(index, "实验", date_from=date(2035, 1, 1)))

    def test_frontmatter_dates_and_tags_are_supported(self):
        self.write("Inbox/front.md", "---\ntitle: 实验计划\ndate: 2035-01-10\ntags: [免疫, RNA]\n---\n实验方案正文")
        hit = self.hits(self.store(), "实验", tags=("rna",))[0]
        self.assertEqual(("实验计划", "2035-01-10"), (hit.title, hit.recorded_at))
        self.assertNotIn("date:", hit.excerpt)

    def test_date_is_converted_to_shanghai_and_requires_matching_marker(self):
        self.write("Inbox/dated.md", note("时区实验", "跨日内容", recorded="2035-01-11T17:00:00+00:00"))
        self.write("Inbox/untrusted.md", note("未匹配", "实验内容").replace("ref=1234567890abcdef", "ref=aaaaaaaaaaaaaaaa"))
        hits = self.hits(self.store(), "实验", date_from=date(2035, 1, 12), date_to=date(2035, 1, 12))
        self.assertEqual(["时区实验"], [hit.title for hit in hits])
        self.assertTrue(hits[0].recorded_at.startswith("2035-01-12"))

    def test_long_paragraphs_are_bounded_and_traceable(self):
        self.write("Inbox/long.md", note("长实验", "实验" * 2300))
        hits = self.hits(self.store(), "实验")
        self.assertGreater(len(hits), 1)
        self.assertTrue(all(0 < len(hit.excerpt) <= 1500 for hit in hits))
        self.assertEqual(1, len({hit.source_ref for hit in hits}))

    def test_incremental_scan_only_reindexes_changed_contents(self):
        path = self.write("Inbox/a.md", "## 实验\n旧内容")
        index = self.store()
        self.assertEqual(1, index.scan().indexed)
        self.assertEqual(0, index.scan().indexed)
        before = path.stat()
        path.write_text("## 实验\n新内容", encoding="utf-8")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(1, index.scan().indexed)
        self.assertFalse(any("旧内容" in hit.excerpt for hit in self.hits(index)))
        self.assertIn("新内容", self.hits(index, "新内容")[0].excerpt)

    def test_rename_delete_and_restart_are_reflected(self):
        path = self.write("Inbox/a.md", note("实验", "重命名记录"))
        index = self.store()
        first = self.hits(index)[0]
        renamed = path.with_name("b.md")
        path.rename(renamed)
        second = self.hits(index)[0]
        self.assertEqual(first.source_id, second.source_id)
        self.assertEqual("Inbox/b.md", second.relative_path)
        index.close()
        reopened = self.store()
        self.assertEqual(second, self.hits(reopened)[0])
        renamed.unlink()
        self.assertEqual((), self.hits(reopened))

    def test_changed_or_deleted_source_invalidates_snapshot(self):
        path = self.write("Inbox/a.md", note("实验", "原文内容"))
        index = self.store()
        snapshot = index.search(NotesQuery("实验"))
        self.assertTrue(index.validate(snapshot))
        path.write_text(note("实验", "改写内容"), encoding="utf-8")
        self.assertFalse(index.validate(snapshot))
        current = index.search(NotesQuery("实验"))
        path.unlink()
        self.assertFalse(index.validate(current))

    def test_mutation_between_scan_and_return_does_not_return_old_body(self):
        path = self.write("Inbox/a.md", note("实验", "扫描时的内容"))
        index = self.store()
        original = index.validate
        def mutate(snapshot):
            path.write_text(note("实验", "后来改写内容"), encoding="utf-8")
            return original(snapshot)
        with patch.object(index, "validate", side_effect=mutate):
            with self.assertRaises(NotesUnavailable):
                index.search(NotesQuery("实验"))


class NotesBoundaryTests(NotesFixture):
    def test_exact_file_allowlist_does_not_expand_to_vault(self):
        self.write("Inbox/allowed.md", "## 允许\n实验")
        self.write("Inbox/other.md", "## 禁止\n实验")
        self.write("Other/also.md", "## 越权\n实验")
        hits = self.hits(self.store(allowed_paths=("Inbox/allowed.md",)), "实验")
        self.assertEqual(["允许"], [hit.title for hit in hits])

    def test_hidden_private_paths_and_private_markers_are_excluded(self):
        self.write("Inbox/visible.md", "## 普通\n实验")
        self.write("Inbox/.hidden.md", "实验隐藏")
        self.write("Inbox/.secret/note.md", "实验隐藏目录")
        self.write("Inbox/PrivateInbox/note.md", "实验私密目录")
        self.write("Inbox/owner-secret/note.md", "实验owner私密")
        self.write("Inbox/partner-secret/note.md", "实验partner私密")
        self.write("Inbox/private-ref.md", "实验\n<!-- wechat-ai-secretary:private-ref=aaaaaaaaaaaaaaaa -->")
        self.write("Inbox/private-frontmatter.md", "---\nprivate: true\n---\n实验")
        hits = self.hits(self.store(private_paths=(self.vault / "Inbox/owner-secret", self.vault / "Inbox/partner-secret")), "实验")
        self.assertEqual(["普通"], [hit.title for hit in hits])

    def test_private_marker_added_after_search_invalidates(self):
        path = self.write("Inbox/a.md", note("普通", "实验"))
        index = self.store()
        snapshot = index.search(NotesQuery())
        path.write_text(path.read_text(encoding="utf-8") + "<!-- wechat-ai-secretary:private-ref=aaaaaaaaaaaaaaaa -->", encoding="utf-8")
        self.assertFalse(index.validate(snapshot))
        self.assertEqual((), self.hits(index))

    def test_private_frontmatter_comments_quotes_duplicates_and_unknown_values(self):
        values = ("private: true # personal", "privacy: private # personal", 'private: "true" # personal', "privacy: 'private'", "private: false\nprivate: true", "private: true\nprivate: false", "private: unclear", "privacy: [public]", "private:")
        for number, value in enumerate(values):
            self.write(f"Inbox/{number}.md", f"---\n{value}\n---\n实验")
        self.write("Inbox/public.md", "---\nprivate: false # normal\nprivacy: 'public'\n---\n## 普通\n实验")
        self.assertEqual(["普通"], [hit.title for hit in self.hits(self.store())])

    def test_unterminated_or_over_budget_frontmatter_is_rejected(self):
        self.write("Inbox/over-budget.md", "---\n" + "description: metadata\n" * 101 + "private: true\n---\n实验")
        self.write("Inbox/unterminated.md", "---\nprivate: false\n实验")
        self.write("Inbox/unknown-over-budget.md", "---\n" + "description: metadata\n" * 101 + "---\n实验")
        self.assertEqual((), self.hits(self.store()))

    def test_symlinks_and_hardlinks_are_rejected(self):
        outside = self.root / "outside.md"
        outside.write_text("## 外部\n实验", encoding="utf-8")
        inbox = self.vault / "Inbox"
        inbox.mkdir()
        os.link(outside, inbox / "hard.md")
        try:
            os.symlink(outside, inbox / "symbolic.md")
        except OSError:
            pass  # Hardlink rejection remains tested without symlink privilege.
        self.assertEqual((), self.hits(self.store()))

    def test_reparse_directory_and_hidden_attribute_are_rejected(self):
        path = self.write("Inbox/reparse/a.md", "实验")
        self.write("Inbox/hidden.md", "实验")
        original = Path.lstat
        def attributes(candidate, *args, **kwargs):
            info = original(candidate, *args, **kwargs)
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            if candidate == path.parent:
                values["st_file_attributes"] = 0x400
            elif candidate.name == "hidden.md":
                values["st_file_attributes"] = 0x2
            return SimpleNamespace(**values)
        index = self.store()
        with patch.object(Path, "lstat", attributes):
            self.assertEqual((), self.hits(index))

    def test_reparse_ancestor_above_vault_is_rejected(self):
        self.write("Inbox/a.md", "实验")
        original = Path.lstat
        def attributes(candidate, *args, **kwargs):
            info = original(candidate, *args, **kwargs)
            if candidate != self.vault.parent:
                return info
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            values["st_file_attributes"] = 0x400
            return SimpleNamespace(**values)
        index = self.store()
        with patch.object(Path, "lstat", attributes):
            self.assertEqual((), self.hits(index))

    def test_private_alias_real_target_is_also_denied(self):
        self.write("Inbox/secret/a.md", "实验私密")
        alias = self.root / "private-alias"
        original = os.path.realpath
        def realpath(value, *args, **kwargs):
            return str(self.vault / "Inbox/secret") if Path(value) == alias else original(value, *args, **kwargs)
        index = self.store(private_paths=(alias,))
        with patch("wechat_secretary.notes_search.os.path.realpath", side_effect=realpath):
            self.assertEqual((), self.hits(index))

    def test_windows_ctime_api_difference_does_not_hide_valid_file(self):
        self.write("Inbox/a.md", "## 实验\n内容")
        index = self.store()
        original = os.fstat
        def fstat(descriptor):
            info = original(descriptor)
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            values["st_ctime_ns"] += 1_000_000
            return SimpleNamespace(**values)
        with patch("wechat_secretary.notes_search.os.fstat", side_effect=fstat):
            self.assertEqual(1, len(self.hits(index)))

    def test_unstable_zero_file_identity_is_rejected(self):
        path = self.write("Inbox/a.md", "实验")
        original = Path.lstat
        def identity(candidate, *args, **kwargs):
            info = original(candidate, *args, **kwargs)
            if candidate != path:
                return info
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            values["st_ino"] = 0
            return SimpleNamespace(**values)
        index = self.store()
        with patch.object(Path, "lstat", identity):
            self.assertEqual((), self.hits(index))

    def test_traversal_absolute_and_hidden_allowlist_entries_fail(self):
        for allowed in (("../outside",), (str(self.root),), (".obsidian",), ("Inbox/../Other",), ()):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                self.store(allowed_paths=allowed)

    def test_configuration_change_invalidates_prior_snapshot_and_data(self):
        self.write("Inbox/a.md", "## A\n实验")
        self.write("Other/b.md", "## B\n实验")
        original = self.store()
        old = original.search(NotesQuery())
        changed = self.store(allowed_paths=("Other",))
        with self.assertRaises(NotesUnavailable):
            original.validate(old)
        self.assertFalse(changed.validate(old))
        self.assertEqual(["B"], [hit.title for hit in self.hits(changed)])

    def test_profile_stores_do_not_share_results_or_exclusion(self):
        self.write("Inbox/a.md", "## 甲\n实验")
        owner = self.store()
        other_vault = self.root / "partner" / "Vault"
        (other_vault / "Inbox").mkdir(parents=True)
        (other_vault / "Inbox" / "b.md").write_text("## 乙\n实验", encoding="utf-8")
        partner = NotesIndex(self.root / "partner" / "index.sqlite3", vault_path=other_vault, allowed_paths=("Inbox",), control_path=self.root / "partner" / "control.json")
        self.addCleanup(partner.close)
        first = self.hits(owner)[0]
        second = self.hits(partner)[0]
        self.assertNotEqual(first.source_id, second.source_id)
        owner.exclude_source(first.source_id)
        self.assertEqual(["乙"], [hit.title for hit in self.hits(partner)])

    def test_per_file_and_total_limits_report_unavailable(self):
        self.write("Inbox/a.md", "a" * 60)
        self.write("Inbox/b.md", "b" * 60)
        index = self.store(max_file_bytes=50)
        with self.assertRaisesRegex(NotesUnavailable, "limit"):
            index.search(NotesQuery())
        index.close()
        total = self.store(max_file_bytes=100, max_total_bytes=100)
        with self.assertRaisesRegex(NotesUnavailable, "limit"):
            total.search(NotesQuery())

    def test_file_count_limit_does_not_return_partial_results(self):
        for number in range(3):
            self.write(f"Inbox/{number}.md", "实验")
        with self.assertRaises(NotesUnavailable):
            self.hits(self.store(max_files=2))

    def test_directory_entry_limit_stops_streaming_before_exhausting_iterator(self):
        (self.vault / "Inbox").mkdir()
        index = self.store(max_files=1)
        visited = []
        vault = self.vault
        class Entries:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def __iter__(self):
                for number in range(1000):
                    visited.append(number)
                    yield SimpleNamespace(path=str(vault / "Inbox" / f"{number}.txt"), is_dir=lambda **kwargs: False)
        with patch("wechat_secretary.notes_search.os.scandir", return_value=Entries()):
            with self.assertRaises(NotesUnavailable):
                index.scan()
        self.assertEqual(21, len(visited))

    def test_permission_failures_never_report_empty_results(self):
        path = self.write("Inbox/a.md", "实验")
        index = self.store()
        original = Path.lstat
        def denied(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError("synthetic permission denied")
            return original(candidate, *args, **kwargs)
        for mock in (patch.object(Path, "lstat", denied), patch("wechat_secretary.notes_search.os.open", side_effect=PermissionError("synthetic permission denied")), patch("wechat_secretary.notes_search.os.scandir", side_effect=PermissionError("synthetic permission denied"))):
            with mock, self.assertRaises(NotesUnavailable):
                index.search(NotesQuery())

    def test_invalid_utf8_reports_unavailable(self):
        path = self.write("Inbox/a.md", "placeholder")
        path.write_bytes(b"\xff\xfe\x00")
        with self.assertRaises(NotesUnavailable):
            self.hits(self.store())

    def test_read_time_mutation_reports_unavailable(self):
        self.write("Inbox/a.md", "实验")
        index = self.store()
        original = os.fstat
        calls = []
        def changed(descriptor):
            info = original(descriptor)
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            calls.append(descriptor)
            values["st_ctime_ns"] += len(calls)
            return SimpleNamespace(**values)
        with patch("wechat_secretary.notes_search.os.fstat", side_effect=changed), self.assertRaises(NotesUnavailable):
            self.hits(index)


class NotesExclusionTests(NotesFixture):
    def test_exclusion_removes_whole_source_and_survives_rebuild_restart(self):
        self.write("Inbox/a.md", note("甲", "第一实验", ref="1111111111111111") + note("乙", "第二实验", ref="2222222222222222"))
        index = self.store()
        snapshot = index.search(NotesQuery())
        result = index.exclude_source(snapshot.hits[0].source_id)
        self.assertEqual("excluded", result.status)
        self.assertFalse(index.validate(snapshot))
        self.assertEqual((), self.hits(index))
        self.assertEqual(0, index.rebuild().indexed)
        self.assertEqual("already_excluded", index.exclude_source(snapshot.hits[0].source_id).status)
        index.close()
        self.assertEqual((), self.hits(self.store()))
        self.assertNotIn("第一实验", self.index_path.read_bytes().decode("utf-8", errors="ignore"))

    def test_exclusion_tracks_rename_edit_and_same_content_copy(self):
        path = self.write("Inbox/a.md", "## 实验\n排除的原文")
        index = self.store()
        hit = self.hits(index)[0]
        index.exclude_source(hit.source_id)
        renamed = path.with_name("renamed.md")
        path.rename(renamed)
        self.write("Inbox/copied.md", "## 实验\n排除的原文")
        self.assertEqual((), self.hits(index))
        renamed.write_text("## 实验\n排除的修订", encoding="utf-8")
        (self.vault / "Inbox/copied.md").write_text("## 实验\n副本的修订", encoding="utf-8")
        self.assertEqual((), self.hits(index))

    def test_same_path_atomic_replacement_remains_excluded(self):
        path = self.write("Inbox/a.md", "## 实验\n原文")
        index = self.store()
        index.exclude_source(self.hits(index)[0].source_id)
        replacement = self.write("Inbox/replacement.tmp", "## 实验\n替换后的内容")
        os.replace(replacement, path)
        self.assertEqual((), self.hits(index))

    def test_new_excluded_version_also_removes_earlier_scanned_copy(self):
        target = self.write("Inbox/z-excluded.md", "## 实验\n最初版本")
        index = self.store()
        index.exclude_source(self.hits(index)[0].source_id)
        replacement = "## 实验\n新版本复制内容"
        target.write_text(replacement, encoding="utf-8")
        self.write("Inbox/a-copy.md", replacement)
        result = index.scan()
        self.assertEqual((0, 2), (result.indexed, result.excluded))
        self.assertEqual((), self.hits(index))
        self.assertNotIn("新版本复制内容", self.index_path.read_bytes().decode("utf-8", errors="ignore"))

    def test_control_contains_no_titles_paths_or_bodies(self):
        self.write("Inbox/秘密文件名.md", "## 秘密标题\n秘密正文")
        index = self.store()
        index.exclude_source(self.hits(index)[0].source_id)
        control = self.control_path.read_text(encoding="utf-8")
        for forbidden in ("秘密", "Inbox", "Vault", "excerpt", "title"):
            self.assertNotIn(forbidden, control)

    def test_old_index_restore_still_honors_exclusion(self):
        self.write("Inbox/a.md", "## 实验\n不应复活")
        index = self.store()
        hit = self.hits(index)[0]
        backup = self.root / "backup.sqlite3"
        shutil.copyfile(self.index_path, backup)
        index.exclude_source(hit.source_id)
        index.close()
        shutil.copyfile(backup, self.index_path)
        restored = self.store()
        self.assertEqual((), self.hits(restored))

    def test_index_deletion_can_rebuild_without_losing_control(self):
        self.write("Inbox/a.md", "## 实验\n不要检索")
        index = self.store()
        index.exclude_source(self.hits(index)[0].source_id)
        control = self.control_path.read_bytes()
        index.close()
        self.index_path.unlink()
        self.assertEqual((), self.hits(self.store()))
        self.assertEqual(control, self.control_path.read_bytes())

    def test_missing_or_corrupt_control_fails_closed(self):
        self.write("Inbox/a.md", "## 实验\n内容")
        index = self.store()
        snapshot = index.search(NotesQuery())
        self.control_path.unlink()
        with self.assertRaises(NotesUnavailable):
            index.validate(snapshot)
        index.close()
        with self.assertRaises(NotesUnavailable):
            self.store()
        self.control_path.write_text("{bad", encoding="utf-8")
        with self.assertRaises(NotesUnavailable):
            self.store()

    def test_control_rollback_is_detected_in_process_and_after_restart(self):
        self.write("Inbox/a.md", "## 实验\n内容")
        index = self.store()
        old_control = self.control_path.read_bytes()
        index.exclude_source(self.hits(index)[0].source_id)
        self.control_path.write_bytes(old_control)
        with self.assertRaises(NotesUnavailable):
            index.current_revision()
        index.close()
        with self.assertRaises(NotesUnavailable):
            self.store()

    def test_control_failure_does_not_claim_exclusion_succeeded(self):
        self.write("Inbox/a.md", "## 实验\n内容")
        index = self.store()
        hit = self.hits(index)[0]
        with patch.object(index, "_write_control", side_effect=OSError("synthetic failure")):
            with self.assertRaises(NotesUnavailable):
                index.exclude_source(hit.source_id)
        self.assertEqual(1, len(self.hits(index)))

    def test_unknown_source_does_not_change_revision(self):
        index = self.store()
        result = index.exclude_source("S-" + "0" * 20)
        self.assertEqual(("not_found", 0), (result.status, result.revision))

    def test_rebuild_forces_refresh_and_invalidates_snapshots_without_losing_exclusions(self):
        self.write("Inbox/a.md", "## 实验\n内容")
        index = self.store()
        snapshot = index.search(NotesQuery())
        control = json.loads(self.control_path.read_text(encoding="utf-8"))
        self.assertEqual(1, index.rebuild().indexed)
        after = json.loads(self.control_path.read_text(encoding="utf-8"))
        self.assertEqual(control["excluded"], after["excluded"])
        self.assertEqual(control["revision"] + 1, after["revision"])
        self.assertFalse(index.validate(snapshot))

    def test_delivery_guard_serializes_exclusion(self):
        self.write("Inbox/a.md", "## 实验\n内容")
        index = self.store()
        snapshot = index.search(NotesQuery())
        started, done = threading.Event(), threading.Event()
        def exclude():
            started.set()
            index.exclude_source(snapshot.hits[0].source_id)
            done.set()
        with index.delivery_guard():
            worker = threading.Thread(target=exclude)
            worker.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(done.wait(0.05))
            self.assertTrue(index.validate(snapshot))
        worker.join(2)
        self.assertTrue(done.is_set())
        self.assertFalse(index.validate(snapshot))

    def test_read_only_state_detection_and_uncertain_database(self):
        self.assertFalse(has_notes_state(self.index_path))
        self.store()
        self.assertTrue(has_notes_state(self.index_path))
        other = self.root / "unrelated.sqlite3"
        connection = sqlite3.connect(other)
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.close()
        self.assertFalse(has_notes_state(other))
        bad = self.root / "broken.sqlite3"
        bad.write_bytes(b"not a database")
        self.assertTrue(has_notes_state(bad))


if __name__ == "__main__":
    unittest.main()
