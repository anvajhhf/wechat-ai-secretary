# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Development setup

Use Windows PowerShell from the repository root:

```powershell
.\scripts\install-local.ps1
$env:PYTHONPATH = "src;runtime/hermes-agent"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The installer downloads the pinned Hermes Agent revision into the ignored
`runtime/` directory. Speech model weights are optional and are not needed for
the default unit tests.

## Pull-request requirements

- Keep changes scoped and add offline regression tests for behavior changes.
- Run the repository secret scan, PowerShell syntax checks, and the full unit
  test suite before opening a pull request.
- Never commit real credentials, account identifiers, messages, recordings,
  Vault content, databases, logs, backups, or downloaded model weights.
- Use synthetic fixtures. Do not make real Weixin, Dida365, DeepSeek, or
  Obsidian writes from automated tests.
- Preserve fail-closed checks around private messages, external writes,
  duplicate delivery, and uncertain network results.
- Document new dependencies and their licenses in `THIRD_PARTY_NOTICES.md`.

By submitting a contribution, you agree that it may be distributed under this
repository's MIT License and that you have the right to submit it.
