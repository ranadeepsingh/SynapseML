#!/usr/bin/env python3
"""
Unified release helper for SynapseML.

End-to-end script to:
- Bump version strings across the repo (with include/exclude rules)
- Snapshot docs into website/versioned_docs and update sidebars/versions.json
- Optionally verify changes (via ripgrep if available) and perform git add/commit/tag/push

Examples
  # Dry-run preview of file changes only
  python tools/release/release.py --from 1.0.15 --to 1.0.16

  # Apply bump + docs snapshot, then commit, tag, and push
  python tools/release/release.py --from 1.0.15 --to 1.0.16 \
      --prev 1.0.15 --apply --git-commit --git-tag --git-push

Requirements: Python 3.11+; only standard library used.
Notes: Respects .gitignore. The bump step and Python verification use
`git check-ignore` to skip ignored files; ripgrep verification also
respects .gitignore by default.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import fnmatch
from functools import lru_cache
from pathlib import Path

# Local helpers (formerly in bump_version.py)
INCLUDE_GLOBS = [
    "README.md",
    "start",
    "docs/**",
    "website/**",
    "tools/docker/**",
    "**/*.md",
    "**/*.js",
    "**/*.ts",
    "**/*.json",
    "**/*.yaml",
]

EXCLUDE_GLOBS = [
    "website/versioned_docs/**",
    "website/versions.json",
    "tools/release/**",
    "node_modules/**",
    ".git/**",
    "**/*.lock",
    "**/*.svg",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.gif",
]

@lru_cache(maxsize=1)
def _load_gitignore_globs(root: Path) -> list[str]:
    gi = root / ".gitignore"
    patterns: list[str] = []
    if not gi.exists():
        return patterns
    try:
        for line in gi.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("/"):
                s = s[1:]
            if s.endswith("/"):
                base = s.rstrip("/")
                patterns.extend([
                    f"{base}",
                    f"{base}/**",
                    f"**/{base}",
                    f"**/{base}/**",
                ])
            else:
                patterns.append(s)
                if "/" not in s:
                    patterns.append(f"**/{s}")
    except Exception:
        return []
    return patterns


def _is_git_ignored(root: Path, path: Path) -> bool:
    git = shutil.which("git")
    rel = path.relative_to(root).as_posix()
    if not git:
        patterns = _load_gitignore_globs(root)
        return any(fnmatch.fnmatch(rel, pat) for pat in patterns)
    try:
        res = subprocess.run([git, "check-ignore", "-q", "--no-index", rel], cwd=root)
        if res.returncode == 0:
            return True
        patterns = _load_gitignore_globs(root)
        return any(fnmatch.fnmatch(rel, pat) for pat in patterns)
    except Exception:
        patterns = _load_gitignore_globs(root)
        return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def collect_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pat in INCLUDE_GLOBS:
        for p in root.glob(pat):
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file():
                        candidates.append(sub)
            else:
                candidates.append(p)

    uniq: list[Path] = []
    seen = set()
    for p in candidates:
        if str(p) in seen:
            continue
        seen.add(str(p))
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, ex) for ex in EXCLUDE_GLOBS):
            continue
        if _is_git_ignored(root, p):
            continue
        uniq.append(p)
    return uniq


def preview_matches(files: list[Path], from_re: re.Pattern[str], root: Path) -> None:
    print("Found occurrences in the following files (excludes frozen docs and .gitignored):")
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="strict")
        except Exception:
            continue
        if from_re.search(txt):
            print(f.as_posix())

    print("\nDry-run: showing sample context (up to first 3 matches per file)\n")
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="strict")
        except Exception:
            continue
        matches = list(from_re.finditer(txt))
        if not matches:
            continue
        print(f"--- {f.as_posix()}")
        lines = txt.splitlines()
        offsets = [m.start() for m in matches[:3]]
        for off in offsets:
            upto = txt[:off]
            line_no = upto.count("\n")
            start = max(0, line_no - 1)
            end = min(len(lines) - 1, line_no + 1)
            for ln in range(start, end + 1):
                print(f"{ln+1}: {lines[ln]}")
        print()


def apply_replacements(files: list[Path], from_re: re.Pattern[str], to: str) -> None:
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="strict")
        except Exception:
            continue
        if not from_re.search(txt):
            continue
        new_txt = from_re.sub(to, txt)
        if new_txt != txt:
            f.write_text(new_txt, encoding="utf-8")
            print(f"Updated: {f.as_posix()}")


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def verify_with_rg(root: Path, from_ver: str, to_ver: str) -> None:
    rg = which("rg")
    if not rg:
        print(
            "ripgrep (rg) not found; skipping rg verification. Use --verify to run Python fallback."
        )
        return
    print("Running ripgrep verification...")
    # Show remaining occurrences of old version (excluding frozen docs and versions.json)
    exclude = [
        "!website/versioned_docs/**",
        "!website/versions.json",
        "!node_modules/**",
    ]
    cmd = [
        rg,
        "-n",
        "-S",
        *sum((["-g", p] for p in exclude), []),
        re.escape(from_ver),
    ]
    subprocess.run(cmd, cwd=root, check=False)
    print("Occurrences of new version:")
    cmd2 = [rg, "-n", "-S", re.escape(to_ver)]
    subprocess.run(cmd2, cwd=root, check=False)


def verify_python(root: Path, from_ver: str, to_ver: str) -> None:
    print("Python verification scanning...")
    from_re = re.compile(re.escape(from_ver))
    to_re = re.compile(re.escape(to_ver))
    files = collect_files(root)
    files = [
        f
        for f in files
        if f.suffix in {".md", ".js", ".ts", ".json", ".yaml"}
        or f.name in {"README.md", "start"}
    ]
    old_hits = []
    new_hits = []
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if from_re.search(txt):
            old_hits.append(f.as_posix())
        if to_re.search(txt):
            new_hits.append(f.as_posix())
    if old_hits:
        print("Remaining old-version occurrences:")
        for p in old_hits[:50]:
            print(p)
        if len(old_hits) > 50:
            print(f"... and {len(old_hits) - 50} more")
    else:
        print("No remaining old-version occurrences in targeted files.")
    print(f"Files containing new version: {len(new_hits)}")


def infer_prev_version(versions_json: Path) -> str | None:
    if not versions_json.exists():
        return None
    try:
        import json

        arr = json.loads(versions_json.read_text(encoding="utf-8"))
        if isinstance(arr, list) and arr:
            return str(arr[0])
    except Exception:
        return None
    return None


def copy_docs_offline(docs_src: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        raise FileExistsError(f"Destination already exists: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=False)
    # Use shutil.copytree (Python 3.8+ supports dirs_exist_ok=False by default)
    import shutil

    shutil.copytree(docs_src, dest_dir, dirs_exist_ok=True)


def copy_sidebars_offline(sidebars_dir: Path, prev: str | None, new: str) -> None:
    import shutil

    new_path = sidebars_dir / f"version-{new}-sidebars.json"
    if prev:
        prev_path = sidebars_dir / f"version-{prev}-sidebars.json"
        if prev_path.exists():
            shutil.copy2(prev_path, new_path)
            print(f"Copied sidebars: {prev_path.name} -> {new_path.name}")
            return
    candidates = sorted(sidebars_dir.glob("version-*-sidebars.json"))
    if candidates:
        shutil.copy2(candidates[-1], new_path)
        print(f"Copied sidebars from {candidates[-1].name}")
    else:
        print(
            "WARNING: No prior versioned sidebars found; consider using docusaurus docs:version"
        )


def update_versions_json(path: Path, new_version: str) -> None:
    import json

    arr = []
    if path.exists():
        try:
            arr = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    if new_version not in arr:
        arr = [new_version] + arr
    path.write_text(json.dumps(arr, indent=2) + "\n", encoding="utf-8")
    print(f"versions.json updated: {arr}")


def run_docusaurus_docs_version(website_dir: Path, new_version: str) -> bool:
    # Try yarn first, then npx
    yarn = which("yarn")
    node = which("node")
    npx = which("npx")
    ok = False
    if yarn and (website_dir / "package.json").exists():
        cmd = [yarn, "run", "docusaurus", "docs:version", new_version]
        print("$", " ".join(cmd))
        res = subprocess.run(cmd, cwd=website_dir)
        ok = res.returncode == 0
    if not ok and npx:
        cmd = [npx, "docusaurus", "docs:version", new_version]
        print("$", " ".join(cmd))
        res = subprocess.run(cmd, cwd=website_dir)
        ok = res.returncode == 0
    if ok:
        print(f"Docusaurus docs:version executed for {new_version}")
    else:
        print("Docusaurus not available or failed; falling back to offline snapshot.")
    return ok


def undo_docs_version(website_dir: Path, version: str) -> None:
    """Best-effort undo of a docs versioning step.
    Removes website/versioned_docs/version-<version>,
    website/versioned_sidebars/version-<version>-sidebars.json,
    and the entry from website/versions.json if present.
    """
    vd = website_dir / "versioned_docs" / f"version-{version}"
    vs = website_dir / "versioned_sidebars" / f"version-{version}-sidebars.json"
    vjson = website_dir / "versions.json"
    if vd.exists():
        print(f"Removing {vd.as_posix()}")
        shutil.rmtree(vd)
    if vs.exists():
        print(f"Removing {vs.as_posix()}")
        try:
            vs.unlink()
        except Exception:
            pass
    if vjson.exists():
        try:
            import json
            arr = json.loads(vjson.read_text(encoding='utf-8'))
            if isinstance(arr, list) and version in arr:
                arr = [v for v in arr if v != version]
                vjson.write_text(json.dumps(arr, indent=2) + "\n", encoding='utf-8')
                print(f"Removed {version} from versions.json")
        except Exception:
            pass


def run_git_steps(
    root: Path,
    to_ver: str,
    do_commit: bool,
    do_tag: bool,
    do_push: bool,
    commit_message: str | None,
    tag_name: str | None,
    tag_message: str | None,
    remote: str,
) -> None:
    git = which("git")
    if not git:
        print("git not found; skipping git operations.")
        return

    def run(cmd):
        print("$", " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)

    if do_commit:
        msg = commit_message or f"Bump SynapseML version: {to_ver}"
        run([git, "add", "-A"])
        # Allow empty commit? Default: no
        subprocess.run([git, "commit", "-m", msg], cwd=root, check=False)

    if do_tag:
        name = tag_name or f"v{to_ver}"
        msg = tag_message or f"SynapseML {to_ver}"
        # Annotated tag
        subprocess.run([git, "tag", "-a", name, "-m", msg], cwd=root, check=False)

    if do_push:
        run([git, "push", remote, "HEAD"])
        run([git, "push", remote, "--tags"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end release helper: bump versions, version docs, and git ops"
    )
    ap.add_argument(
        "--from", dest="from_version", required=True, help="Current version string"
    )
    ap.add_argument("--to", dest="to_version", required=True, help="New version string")
    ap.add_argument(
        "--prev",
        dest="prev_version",
        default=None,
        help="Previous version for sidebars inference",
    )
    ap.add_argument(
        "--apply", action="store_true", help="Apply edits; otherwise dry-run preview"
    )
    ap.add_argument(
        "--undo-docs", action="store_true", help="Undo docs versioning for --to version"
    )
    ap.add_argument(
        "--root", default=str(Path.cwd()), help="Repository root (default: CWD)"
    )

    # Scope control
    ap.add_argument(
        "--skip-bump", action="store_true", help="Skip version string bumping"
    )
    ap.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip docs snapshot and versions.json update",
    )

    # Verification
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Verify results (uses rg if available, else Python)",
    )

    # Git operations
    ap.add_argument("--git-commit", action="store_true", help="Run git add/commit")
    ap.add_argument("--git-tag", action="store_true", help="Create annotated git tag")
    ap.add_argument(
        "--git-push", action="store_true", help="Push HEAD and tags to remote"
    )
    ap.add_argument("--commit-message", default=None, help="Custom commit message")
    ap.add_argument("--tag-name", default=None, help="Custom tag name (default v<to>)")
    ap.add_argument("--tag-message", default=None, help="Custom tag message")
    ap.add_argument(
        "--remote", default="origin", help="Git remote to push to (default origin)"
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Bump versions
    if not args.skip_bump:
        from_re = re.compile(re.escape(args.from_version))
        files = collect_files(root)
        files = [
            f
            for f in files
            if f.suffix in {".md", ".js", ".ts", ".json", ".yaml"}
            or f.name in {"README.md", "start"}
        ]
        if not args.apply:
            preview_matches(files, from_re, root)
        else:
            apply_replacements(files, from_re, args.to_version)
            print(f"Version bump complete: {args.from_version} -> {args.to_version}")

    # Docs versioning
    if args.undo_docs:
        website_dir = root / "website"
        if not website_dir.is_dir():
            print("Undo skipped: missing website/ directory")
        else:
            undo_docs_version(website_dir, args.to_version)
    elif not args.skip_docs and args.apply:
        website_dir = root / "website"
        docs_src = root / "docs"
        dest_dir = website_dir / "versioned_docs" / f"version-{args.to_version}"
        sidebars_dir = website_dir / "versioned_sidebars"
        versions_json = website_dir / "versions.json"
        if not website_dir.is_dir() or not docs_src.is_dir():
            print("Skipping docs snapshot: missing website/ or docs/ directory")
        else:
            # Prefer docusaurus; fallback to offline
            used_docusaurus = run_docusaurus_docs_version(website_dir, args.to_version)
            if not used_docusaurus:
                prev = args.prev_version or infer_prev_version(versions_json)
                copy_docs_offline(docs_src, dest_dir)
                print(f"Created versioned docs at {dest_dir.as_posix()}")
                copy_sidebars_offline(sidebars_dir, prev, args.to_version)
                update_versions_json(versions_json, args.to_version)
            print(f"Docs versioning complete for {args.to_version}")
    elif not args.skip_docs and not args.apply:
        website_dir = root / "website"
        dest_dir = website_dir / "versioned_docs" / f"version-{args.to_version}"
        sidebars_json = website_dir / "versioned_sidebars" / f"version-{args.to_version}-sidebars.json"
        print("Dry-run: would version docs (no changes):")
        print(f"  create {dest_dir.as_posix()}")
        print(f"  create {sidebars_json.as_posix()}")
        print(f"  update { (website_dir / 'versions.json').as_posix() }")

    # Verification
    if args.verify:
        if which("rg"):
            verify_with_rg(root, args.from_version, args.to_version)
        else:
            verify_python(root, args.from_version, args.to_version)

    # Git operations
    if args.git_commit or args.git_tag or args.git_push:
        run_git_steps(
            root,
            args.to_version,
            args.git_commit,
            args.git_tag,
            args.git_push,
            args.commit_message,
            args.tag_name,
            args.tag_message,
            args.remote,
        )


if __name__ == "__main__":
    main()
