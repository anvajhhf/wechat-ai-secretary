# Third-party notices

This repository contains integration code and small compatibility patches for
third-party software. Runtime credentials, user data, downloaded model weights,
and installed dependency packages are not part of this repository.

## Hermes Agent

The compatibility patches under `patches/` target
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent),
which is distributed under the MIT License:

> MIT License
>
> Copyright (c) 2025 Nous Research
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Python runtime dependencies

The project declares dependencies but does not redistribute their source or
binary packages. Review the license shipped with the exact installed version.

| Component | Declared license | Project |
| --- | --- | --- |
| Pillow | MIT-CMU | <https://python-pillow.org/> |
| faster-whisper | MIT | <https://github.com/SYSTRAN/faster-whisper> |
| silk-python | BSD | <https://github.com/synodriver/pysilk> |
| sherpa-onnx / sherpa-onnx-core | Apache-2.0 | <https://github.com/k2-fsa/sherpa-onnx> |
| tzdata | Apache-2.0 | <https://github.com/python/tzdata> |

Hermes Agent has its own transitive dependency set. Its installation metadata
and license files remain authoritative for those packages.

## Speech model downloads

Speech model weights are downloaded separately into the ignored `runtime/`
directory and are not covered by this repository's MIT License. Before
downloading or redistributing a model, review the license and model card at the
pinned source revision:

- [Paraformer export](https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28)
- [SenseVoice export](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17)
- [Whisper](https://github.com/openai/whisper)

## External services and trademarks

DeepSeek, Weixin/WeChat, 滴答清单/Dida365, Obsidian, and their marks belong to
their respective owners. This is an unofficial community project and is not
endorsed by or affiliated with those service providers. Their current terms,
API policies, and account requirements apply independently.
