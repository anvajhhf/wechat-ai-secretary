from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositorySafetyTests(unittest.TestCase):
    def test_license_and_security_documents_are_present(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 anvajhhf", license_text)
        self.assertIn("Copyright (c) 2025 Nous Research", notices)
        self.assertIn("security/advisories/new", security)

    def test_project_metadata_declares_public_license_and_urls(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = metadata["project"]

        self.assertEqual({"file": "LICENSE"}, project["license"])
        self.assertEqual(
            "https://github.com/anvajhhf/wechat-ai-secretary",
            project["urls"]["Repository"],
        )

    def test_all_remote_actions_are_pinned_to_full_commit_hashes(self) -> None:
        workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
        self.assertTrue(workflows)
        for workflow in workflows:
            text = workflow.read_text(encoding="utf-8")
            references = re.findall(
                r"^\s*(?:-\s+)?uses:\s*([^@\s]+)@([^\s#]+)",
                text,
                re.MULTILINE,
            )
            with self.subTest(workflow=workflow.name):
                self.assertTrue(references)
                for action, revision in references:
                    self.assertRegex(
                        action,
                        r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$",
                    )
                    self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_ci_fetches_history_and_public_codeql_is_gated(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        codeql = (ROOT / ".github" / "workflows" / "codeql.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("fetch-depth: 0", ci)
        self.assertIn("github.event.repository.private == false", codeql)
        self.assertIn("security-events: write", codeql)

    def test_secret_scan_covers_current_tree_and_reachable_history(self) -> None:
        scanner = (ROOT / "scripts" / "check-repository-secrets.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("git ls-files --cached --others --exclude-standard", scanner)
        self.assertIn("git rev-list --objects --all", scanner)
        self.assertIn("git log --all", scanner)
        self.assertIn("GitHub 访问令牌", scanner)
        self.assertIn("本机用户目录", scanner)

    def test_local_profiles_and_runtime_artifacts_are_ignored(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

        for rule in (
            "config/secretary*.toml",
            "!config/secretary.example.toml",
            "runtime/",
            "*.wasbak",
            "*.sqlite",
            "*.onnx",
            "*.wav",
            "*.png",
        ):
            self.assertIn(rule, ignore)

    def test_public_readme_contains_no_personal_absolute_paths(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("D:\\Codex", readme)
        self.assertNotIn("D:\\WeChatAIData", readme)
        self.assertIn("<data-root>", readme)
        self.assertIn("SECURITY.md", readme)

    def test_dependabot_covers_python_and_actions(self) -> None:
        dependabot = (ROOT / ".github" / "dependabot.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("package-ecosystem: pip", dependabot)
        self.assertIn("package-ecosystem: github-actions", dependabot)


if __name__ == "__main__":
    unittest.main()
