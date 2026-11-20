"""SessionStart hook: sync the repo and show who is working on what.

Claude Code runs this script when a session starts and passes its output to Claude as context. It
fetches the remote and fast-forwards `main` when `main` is checked out (it never merges or
overwrites local work). Then it prints the "Now" and "Final deliverables" sections of
docs/STATUS.md as they are on `origin/main`, whatever branch is checked out, and the open GitHub
issues with their assignees. It uses only the standard library and always exits 0, so a failure
never blocks a session.

Usage: python session_start.py [path-to-timenet-3w]
Without an argument, the repo is the folder two levels above this script.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

SECTIONS = ("Now", "Final deliverables")


def run(args: list[str], cwd: Path) -> tuple[int, str]:
    """Run a command and return its exit code and combined output."""
    try:
        done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30, encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, str(error)
    return done.returncode, (done.stdout + done.stderr).strip()


def sync(repo: Path) -> str:
    """Fetch the remote, fast-forward main if it is checked out, and describe the result."""
    code, output = run(["git", "fetch", "--quiet", "origin"], repo)
    if code != 0:
        return f"git fetch FAILED, the status below may be old. Tell the user.\n{output}"
    _, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
    if branch == "main":
        code, output = run(["git", "merge", "--ff-only", "--quiet", "origin/main"], repo)
        if code == 0:
            return "On main: fast-forwarded to origin/main."
        return f"On main: fast-forward FAILED, nothing was changed. Tell the user before working.\n{output}"
    _, behind = run(["git", "rev-list", "--count", f"{branch}..origin/main"], repo)
    return (
        f"On branch {branch} (not main): it was not updated. origin/main has {behind} commit(s) it lacks;"
        " merge or rebase onto origin/main before opening the PR."
    )


def status_sections(repo: Path) -> str:
    """Return the sections of docs/STATUS.md on origin/main listed in SECTIONS."""
    code, text = run(["git", "show", "origin/main:docs/STATUS.md"], repo)
    if code != 0:
        path = repo / "docs" / "STATUS.md"
        if not path.exists():
            return "docs/STATUS.md not found."
        text = path.read_text(encoding="utf-8")
    parts = []
    for title in SECTIONS:
        match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
        if match:
            parts.append(f"## {title}\n{match.group(1).strip()}")
    return "\n\n".join(parts) or "docs/STATUS.md has no Now or Final deliverables section."


def open_issues(repo: Path) -> str:
    """Return the open GitHub issues with their assignees, if gh is available."""
    code, output = run(
        ["gh", "issue", "list", "--state", "open", "--limit", "50", "--json", "number,title,assignees"], repo
    )
    if code != 0:
        return f"Could not list the GitHub issues (is gh installed and logged in?): {output[:200]}"
    issues = json.loads(output or "[]")
    if not issues:
        return "No open issues."
    lines = []
    for issue in sorted(issues, key=lambda item: item["number"]):
        owners = ", ".join(person["login"] for person in issue["assignees"]) or "UNASSIGNED"
        lines.append(f"#{issue['number']} [{owners}] {issue['title']}")
    return "\n".join(lines)


def main() -> None:
    """Print the session context for Claude."""
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    print(f"# timenet-3w session context ({repo})")
    print(sync(repo))
    print()
    print(status_sections(repo))
    print()
    print("## Open GitHub issues (claim one before working on it, see CLAUDE.md)")
    print(open_issues(repo))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - a hook must never block a session
        print(f"session_start hook error: {error}")
    sys.exit(0)
