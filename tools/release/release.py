#!/usr/bin/env python3
"""
Unified release helper for SynapseML.

End-to-end script to:
- Bump version strings across the repo (with include/exclude rules)
- Update build configuration for Spark variants (build.sbt, PackageUtils.scala, etc.)
- Snapshot docs into website/versioned_docs and update sidebars/versions.json
- Optionally verify changes (via ripgrep if available) and perform git add/commit/tag/push

Examples
  # Dry-run preview of file changes only
  python tools/release/release.py version --from 1.1.0 --to 1.1.1

  # Apply bump + docs snapshot, then commit, tag, and push
  python tools/release/release.py version --from 1.1.0 --to 1.1.1 \
      --apply --verify --git-commit --git-tag --git-push

  # Spark 4.0 variant release (updates build.sbt, creates v1.1.1-spark4.0 tag)
  python tools/release/release.py version --from 1.1.0 --to 1.1.1 \
      --spark-variant spark4.0 --apply --git-commit --git-tag --git-push

Requirements: Python 3.11+; only standard library used (PyYAML optional).
Notes: Respects .gitignore. The bump step and Python verification use
`git check-ignore` to skip ignored files; ripgrep verification also
respects .gitignore by default.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import fnmatch
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

# ============================================================================
# Spark Profile Configuration
# ============================================================================

@dataclass
class SparkProfile:
    """Configuration for a Spark variant release."""
    spark_version: str
    scala_version: str
    scala_major: str
    java_version: str
    python_version: str
    pyspark_version: str
    pyarrow_version: str
    hadoop_version: str
    isolation_forest_spark: str
    isolation_forest_version: str = "3.0.5"
    branch: str = "master"
    tag_suffix: str = ""
    extra_conda_deps: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SparkProfile":
        return cls(
            spark_version=data["spark_version"],
            scala_version=data["scala_version"],
            scala_major=data["scala_major"],
            java_version=data["java_version"],
            python_version=data["python_version"],
            pyspark_version=data["pyspark_version"],
            pyarrow_version=data["pyarrow_version"],
            hadoop_version=data["hadoop_version"],
            isolation_forest_spark=data["isolation_forest_spark"],
            isolation_forest_version=data.get("isolation_forest_version", "3.0.5"),
            branch=data.get("branch", "master"),
            tag_suffix=data.get("tag_suffix", ""),
            extra_conda_deps=data.get("extra_conda_deps", []),
        )


def load_spark_profiles(path: Path) -> dict[str, SparkProfile]:
    """Load Spark profiles from YAML configuration."""
    if not path.exists():
        print(f"WARNING: Spark profiles not found at {path}")
        return {}

    content = path.read_text(encoding="utf-8")

    # Try YAML first (if PyYAML available), fallback to simple parsing
    try:
        import yaml
        data = yaml.safe_load(content)
    except ImportError:
        # Simple YAML-like parsing for basic cases
        data = _parse_simple_yaml(content)

    profiles = {}
    for name, config in data.items():
        if isinstance(config, dict):
            profiles[name] = SparkProfile.from_dict(config)

    return profiles


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    """Simple YAML parser for basic key-value configs (fallback when PyYAML not available)."""
    result = {}
    current_profile = None
    current_dict = {}

    for line in content.splitlines():
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check for profile name (no leading spaces, ends with :)
        if not line.startswith(" ") and line.rstrip().endswith(":"):
            # Save previous profile
            if current_profile and current_dict:
                result[current_profile] = current_dict
            current_profile = line.rstrip(":").strip()
            current_dict = {}
        elif current_profile and ":" in line:
            # Key-value pair
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                current_dict[key] = value

    # Save last profile
    if current_profile and current_dict:
        result[current_profile] = current_dict

    return result


def detect_current_profile(root: Path) -> SparkProfile | None:
    """Detect current Spark profile from build.sbt."""
    build_sbt = root / "build.sbt"
    if not build_sbt.exists():
        return None

    content = build_sbt.read_text(encoding="utf-8")

    # Extract current versions
    spark_match = re.search(r'val sparkVersion = "([^"]+)"', content)
    scala_match = re.search(r'scalaVersion := "([^"]+)"', content)
    scala_major_match = re.search(r'val scalaMajorVersion = ([0-9.]+)', content)

    if not spark_match or not scala_match:
        return None

    scala_version = scala_match.group(1)
    scala_major = scala_major_match.group(1) if scala_major_match else scala_version.rsplit(".", 1)[0]

    return SparkProfile(
        spark_version=spark_match.group(1),
        scala_version=scala_version,
        scala_major=scala_major,
        java_version="11",  # Default
        python_version="3.11",
        pyspark_version=spark_match.group(1),
        pyarrow_version="10.0.1",
        hadoop_version="3",
        isolation_forest_spark=spark_match.group(1),
    )


# ============================================================================
# File Collection and Filtering
# ============================================================================

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
    "target/**",
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


# ============================================================================
# Pre-flight Checks
# ============================================================================

class PreflightChecker:
    """Run safety checks before applying changes."""

    def __init__(self, root: Path, from_ver: str, to_ver: str, spark_variant: str | None = None):
        self.root = root
        self.from_ver = from_ver
        self.to_ver = to_ver
        self.spark_variant = spark_variant
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check_git_clean(self) -> bool:
        """Ensure working directory is clean."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.root,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            self.warnings.append("Git working directory has uncommitted changes")
            return False
        return True

    def check_branch(self, expected: str) -> bool:
        """Verify we're on the expected branch."""
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.root,
            capture_output=True,
            text=True
        )
        current = result.stdout.strip()
        if current != expected:
            self.warnings.append(f"On branch '{current}', expected '{expected}'")
            return False
        return True

    def check_version_exists(self) -> bool:
        """Check if version string exists in expected files."""
        readme = self.root / "README.md"
        if not readme.exists():
            self.errors.append("README.md not found")
            return False

        content = readme.read_text(encoding="utf-8")
        if self.from_ver not in content:
            self.errors.append(f"Version {self.from_ver} not found in README.md")
            return False
        return True

    def check_tag_not_exists(self, tag_name: str) -> bool:
        """Ensure tag doesn't already exist."""
        result = subprocess.run(
            ["git", "tag", "-l", tag_name],
            cwd=self.root,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            self.errors.append(f"Tag {tag_name} already exists")
            return False
        return True

    def run_all(self, tag_name: str | None = None) -> tuple[bool, list[str], list[str]]:
        """Run all pre-flight checks."""
        print("\n" + "=" * 60)
        print("Pre-flight Checks")
        print("=" * 60)

        all_passed = True

        # Git clean check
        passed = self.check_git_clean()
        status = "PASS" if passed else "WARN"
        print(f"[{status}] Git working directory clean")

        # Version exists check
        passed = self.check_version_exists()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Version {self.from_ver} found in README.md")
        if not passed:
            all_passed = False

        # Tag check
        if tag_name:
            passed = self.check_tag_not_exists(tag_name)
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] Tag {tag_name} does not exist")
            if not passed:
                all_passed = False

        print()
        return all_passed, self.errors, self.warnings


# ============================================================================
# Build Configuration Updates (Spark Variants)
# ============================================================================

def update_build_sbt(
    path: Path,
    current: SparkProfile,
    target: SparkProfile,
    dry_run: bool = True
) -> list[str]:
    """Update build.sbt with new Spark/Scala versions."""
    if not path.exists():
        return []

    changes = []
    content = path.read_text(encoding="utf-8")
    original = content

    # Spark version
    pattern = re.compile(r'(val sparkVersion = ")[^"]+(")')
    if pattern.search(content) and current.spark_version != target.spark_version:
        content = pattern.sub(rf'\g<1>{target.spark_version}\g<2>', content)
        changes.append(f"sparkVersion: {current.spark_version} -> {target.spark_version}")

    # Scala version
    pattern = re.compile(r'(scalaVersion := ")[^"]+(")')
    if pattern.search(content) and current.scala_version != target.scala_version:
        content = pattern.sub(rf'\g<1>{target.scala_version}\g<2>', content)
        changes.append(f"scalaVersion: {current.scala_version} -> {target.scala_version}")

    # Scala major version
    pattern = re.compile(r'(val scalaMajorVersion = )[0-9.]+')
    if pattern.search(content) and current.scala_major != target.scala_major:
        content = pattern.sub(rf'\g<1>{target.scala_major}', content)
        changes.append(f"scalaMajorVersion: {current.scala_major} -> {target.scala_major}")

    # Isolation forest dependency
    pattern = re.compile(r'(isolation-forest_)[0-9.]+')
    if pattern.search(content) and current.isolation_forest_spark != target.isolation_forest_spark:
        content = pattern.sub(rf'\g<1>{target.isolation_forest_spark}', content)
        changes.append(f"isolation-forest: {current.isolation_forest_spark} -> {target.isolation_forest_spark}")

    # Isolation forest version (if different)
    if current.isolation_forest_version != target.isolation_forest_version:
        old_pattern = re.compile(rf'(isolation-forest_[0-9.]+" % "){current.isolation_forest_version}(")')
        if old_pattern.search(content):
            content = old_pattern.sub(rf'\g<1>{target.isolation_forest_version}\g<2>', content)
            changes.append(f"isolation-forest version: {current.isolation_forest_version} -> {target.isolation_forest_version}")

    if not dry_run and content != original:
        path.write_text(content, encoding="utf-8")

    return changes


def update_package_utils(
    path: Path,
    current: SparkProfile,
    target: SparkProfile,
    dry_run: bool = True
) -> list[str]:
    """Update PackageUtils.scala with new Spark/Scala versions."""
    if not path.exists():
        return []

    changes = []
    content = path.read_text(encoding="utf-8")
    original = content

    # spark-avro coordinate pattern
    pattern = re.compile(rf'(spark-avro_){current.scala_major}:{current.spark_version}')
    if pattern.search(content):
        content = pattern.sub(rf'\g<1>{target.scala_major}:{target.spark_version}', content)
        changes.append(f"spark-avro: {current.scala_major}:{current.spark_version} -> {target.scala_major}:{target.spark_version}")

    if not dry_run and content != original:
        path.write_text(content, encoding="utf-8")

    return changes


def update_environment_yml(
    path: Path,
    current: SparkProfile,
    target: SparkProfile,
    dry_run: bool = True
) -> list[str]:
    """Update environment.yml with new Python/Spark versions."""
    if not path.exists():
        return []

    changes = []
    content = path.read_text(encoding="utf-8")
    original = content

    # PySpark version
    if current.pyspark_version != target.pyspark_version:
        pattern = re.compile(rf'(pyspark==){current.pyspark_version}')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.pyspark_version}', content)
            changes.append(f"pyspark: {current.pyspark_version} -> {target.pyspark_version}")

    # PyArrow version
    if current.pyarrow_version != target.pyarrow_version:
        pattern = re.compile(rf'(pyarrow==){current.pyarrow_version}')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.pyarrow_version}', content)
            changes.append(f"pyarrow: {current.pyarrow_version} -> {target.pyarrow_version}")

    # Python version
    if current.python_version != target.python_version:
        pattern = re.compile(rf'(python=){current.python_version}')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.python_version}', content)
            changes.append(f"python: {current.python_version} -> {target.python_version}")

    if not dry_run and content != original:
        path.write_text(content, encoding="utf-8")

    return changes


def update_dockerfile(
    path: Path,
    current: SparkProfile,
    target: SparkProfile,
    dry_run: bool = True
) -> list[str]:
    """Update Dockerfile with new Spark version."""
    if not path.exists():
        return []

    changes = []
    content = path.read_text(encoding="utf-8")
    original = content

    # SPARK_VERSION
    if current.spark_version != target.spark_version:
        pattern = re.compile(rf'(SPARK_VERSION=){current.spark_version}')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.spark_version}', content)
            changes.append(f"SPARK_VERSION: {current.spark_version} -> {target.spark_version}")

        # Also handle ENV form
        pattern = re.compile(rf'(ENV SPARK_VERSION=){current.spark_version}')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.spark_version}', content)

    # HADOOP_VERSION
    if current.hadoop_version != target.hadoop_version:
        pattern = re.compile(rf'(HADOOP_VERSION=){current.hadoop_version}')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.hadoop_version}', content)
            changes.append(f"HADOOP_VERSION: {current.hadoop_version} -> {target.hadoop_version}")

    if not dry_run and content != original:
        path.write_text(content, encoding="utf-8")

    return changes


def update_start_script(
    path: Path,
    current: SparkProfile,
    target: SparkProfile,
    dry_run: bool = True
) -> list[str]:
    """Update start script with new Spark version."""
    if not path.exists():
        return []

    changes = []
    content = path.read_text(encoding="utf-8")
    original = content

    # SPARK_VERSION
    if current.spark_version != target.spark_version:
        pattern = re.compile(rf'(SPARK_VERSION="){current.spark_version}(")')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.spark_version}\g<2>', content)
            changes.append(f"SPARK_VERSION: {current.spark_version} -> {target.spark_version}")

    # HADOOP_VERSION
    if current.hadoop_version != target.hadoop_version:
        pattern = re.compile(rf'(HADOOP_VERSION="){current.hadoop_version}(")')
        if pattern.search(content):
            content = pattern.sub(rf'\g<1>{target.hadoop_version}\g<2>', content)
            changes.append(f"HADOOP_VERSION: {current.hadoop_version} -> {target.hadoop_version}")

    if not dry_run and content != original:
        path.write_text(content, encoding="utf-8")

    return changes


def apply_build_config_updates(
    root: Path,
    current: SparkProfile,
    target: SparkProfile,
    dry_run: bool = True,
    skip_docker: bool = False
) -> dict[str, list[str]]:
    """Apply all build configuration updates for a Spark variant."""
    all_changes = {}

    # build.sbt
    build_sbt = root / "build.sbt"
    changes = update_build_sbt(build_sbt, current, target, dry_run)
    if changes:
        all_changes["build.sbt"] = changes

    # PackageUtils.scala
    pkg_utils = root / "core" / "src" / "main" / "scala" / "com" / "microsoft" / "azure" / "synapse" / "ml" / "core" / "env" / "PackageUtils.scala"
    changes = update_package_utils(pkg_utils, current, target, dry_run)
    if changes:
        all_changes["PackageUtils.scala"] = changes

    # environment.yml
    env_yml = root / "environment.yml"
    changes = update_environment_yml(env_yml, current, target, dry_run)
    if changes:
        all_changes["environment.yml"] = changes

    if not skip_docker:
        # Dockerfiles
        for dockerfile in [
            root / "tools" / "docker" / "demo" / "Dockerfile",
            root / "tools" / "docker" / "minimal" / "Dockerfile",
        ]:
            changes = update_dockerfile(dockerfile, current, target, dry_run)
            if changes:
                all_changes[dockerfile.relative_to(root).as_posix()] = changes

        # start script
        start = root / "start"
        changes = update_start_script(start, current, target, dry_run)
        if changes:
            all_changes["start"] = changes

    return all_changes


# ============================================================================
# Version String Updates
# ============================================================================

def preview_matches(files: list[Path], from_re: re.Pattern[str], root: Path) -> None:
    print("Found occurrences in the following files (excludes frozen docs and .gitignored):")
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="strict")
        except Exception:
            continue
        if from_re.search(txt):
            print(f"  {f.relative_to(root).as_posix()}")

    print("\nDry-run: showing sample context (up to first 3 matches per file)\n")
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="strict")
        except Exception:
            continue
        matches = list(from_re.finditer(txt))
        if not matches:
            continue
        print(f"--- {f.relative_to(root).as_posix()}")
        lines = txt.splitlines()
        offsets = [m.start() for m in matches[:3]]
        for off in offsets:
            upto = txt[:off]
            line_no = upto.count("\n")
            start = max(0, line_no - 1)
            end = min(len(lines) - 1, line_no + 1)
            for ln in range(start, end + 1):
                print(f"  {ln+1}: {lines[ln]}")
        print()


def apply_replacements(files: list[Path], from_re: re.Pattern[str], to: str, root: Path) -> int:
    count = 0
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
            print(f"  Updated: {f.relative_to(root).as_posix()}")
            count += 1
    return count


# ============================================================================
# Verification
# ============================================================================

def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def verify_with_rg(root: Path, from_ver: str, to_ver: str) -> None:
    rg = which("rg")
    if not rg:
        print("ripgrep (rg) not found; using Python verification.")
        verify_python(root, from_ver, to_ver)
        return

    print("Running ripgrep verification...")
    exclude = [
        "!website/versioned_docs/**",
        "!website/versions.json",
        "!node_modules/**",
        "!tools/release/**",
    ]
    cmd = [
        rg, "-n", "-S",
        *sum((["-g", p] for p in exclude), []),
        re.escape(from_ver),
    ]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.stdout.strip():
        print("Remaining old-version occurrences:")
        print(result.stdout)
    else:
        print("No remaining old-version occurrences found.")

    print("\nNew version occurrences:")
    cmd2 = [rg, "-c", "-S", re.escape(to_ver)]
    result2 = subprocess.run(cmd2, cwd=root, capture_output=True, text=True)
    lines = result2.stdout.strip().splitlines()
    print(f"  Found in {len(lines)} files")


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
            old_hits.append(f.relative_to(root).as_posix())
        if to_re.search(txt):
            new_hits.append(f.relative_to(root).as_posix())

    if old_hits:
        print("Remaining old-version occurrences:")
        for p in old_hits[:20]:
            print(f"  {p}")
        if len(old_hits) > 20:
            print(f"  ... and {len(old_hits) - 20} more")
    else:
        print("No remaining old-version occurrences in targeted files.")

    print(f"Files containing new version: {len(new_hits)}")


# ============================================================================
# Documentation Versioning
# ============================================================================

def infer_prev_version(versions_json: Path) -> str | None:
    if not versions_json.exists():
        return None
    try:
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
    shutil.copytree(docs_src, dest_dir, dirs_exist_ok=True)


def copy_sidebars_offline(sidebars_dir: Path, prev: str | None, new: str) -> None:
    new_path = sidebars_dir / f"version-{new}-sidebars.json"
    if prev:
        prev_path = sidebars_dir / f"version-{prev}-sidebars.json"
        if prev_path.exists():
            shutil.copy2(prev_path, new_path)
            print(f"  Copied sidebars: {prev_path.name} -> {new_path.name}")
            return
    candidates = sorted(sidebars_dir.glob("version-*-sidebars.json"))
    if candidates:
        shutil.copy2(candidates[-1], new_path)
        print(f"  Copied sidebars from {candidates[-1].name}")
    else:
        print("  WARNING: No prior versioned sidebars found")


def update_versions_json(path: Path, new_version: str) -> None:
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
    print(f"  versions.json updated: {arr[:5]}...")


def run_docusaurus_docs_version(website_dir: Path, new_version: str) -> bool:
    yarn = which("yarn")
    npx = which("npx")
    ok = False

    if yarn and (website_dir / "package.json").exists():
        cmd = [yarn, "run", "docusaurus", "docs:version", new_version]
        print("  $", " ".join(cmd))
        res = subprocess.run(cmd, cwd=website_dir, capture_output=True)
        ok = res.returncode == 0

    if not ok and npx:
        cmd = [npx, "docusaurus", "docs:version", new_version]
        print("  $", " ".join(cmd))
        res = subprocess.run(cmd, cwd=website_dir, capture_output=True)
        ok = res.returncode == 0

    if ok:
        print(f"  Docusaurus docs:version executed for {new_version}")
    else:
        print("  Docusaurus not available; using offline snapshot.")

    return ok


def undo_docs_version(website_dir: Path, version: str) -> None:
    """Undo a docs versioning step."""
    vd = website_dir / "versioned_docs" / f"version-{version}"
    vs = website_dir / "versioned_sidebars" / f"version-{version}-sidebars.json"
    vjson = website_dir / "versions.json"

    if vd.exists():
        print(f"  Removing {vd.as_posix()}")
        shutil.rmtree(vd)

    if vs.exists():
        print(f"  Removing {vs.as_posix()}")
        vs.unlink()

    if vjson.exists():
        try:
            arr = json.loads(vjson.read_text(encoding="utf-8"))
            if isinstance(arr, list) and version in arr:
                arr = [v for v in arr if v != version]
                vjson.write_text(json.dumps(arr, indent=2) + "\n", encoding="utf-8")
                print(f"  Removed {version} from versions.json")
        except Exception:
            pass


# ============================================================================
# Git Operations
# ============================================================================

def run_git_steps(
    root: Path,
    to_ver: str,
    spark_variant: str | None,
    profiles: dict[str, SparkProfile],
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
        print("ERROR: git not found")
        return

    def run(cmd, check=True):
        print("  $", " ".join(cmd))
        return subprocess.run(cmd, cwd=root, check=check)

    # Determine tag name based on variant
    if not tag_name:
        if spark_variant and spark_variant in profiles:
            suffix = profiles[spark_variant].tag_suffix
            tag_name = f"v{to_ver}{suffix}"
        else:
            tag_name = f"v{to_ver}"

    print("\n" + "=" * 60)
    print("Git Operations")
    print("=" * 60)

    if do_commit:
        variant_info = f" ({spark_variant})" if spark_variant else ""
        msg = commit_message or f"Release SynapseML {to_ver}{variant_info}"
        run([git, "add", "-A"])
        result = subprocess.run(
            [git, "commit", "-m", msg],
            cwd=root,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  Committed: {msg}")
        elif "nothing to commit" in result.stdout + result.stderr:
            print("  No changes to commit")
        else:
            print(f"  Commit failed: {result.stderr}")

    if do_tag:
        variant_info = f" ({spark_variant})" if spark_variant else ""
        msg = tag_message or f"SynapseML {to_ver}{variant_info}"

        # Check if tag exists
        result = subprocess.run(
            [git, "tag", "-l", tag_name],
            cwd=root,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(f"  WARNING: Tag {tag_name} already exists, skipping")
        else:
            run([git, "tag", "-a", tag_name, "-m", msg], check=False)
            print(f"  Created tag: {tag_name}")

    if do_push:
        run([git, "push", remote, "HEAD"])
        run([git, "push", remote, "--tags"])


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SynapseML Release Automation - Bump versions, update build configs, version docs, and git ops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run preview
  python tools/release/release.py version --from 1.1.0 --to 1.1.1

  # Standard release (Spark 3.5)
  python tools/release/release.py version --from 1.1.0 --to 1.1.1 \\
      --apply --verify --git-commit --git-tag --git-push

  # Spark 4.0 variant release
  python tools/release/release.py version --from 1.1.0 --to 1.1.1 \\
      --spark-variant spark4.0 --apply --git-commit --git-tag --git-push
"""
    )

    # Subcommand (for future extensibility)
    subparsers = ap.add_subparsers(dest="command", help="Command to run")

    # Version command
    version_parser = subparsers.add_parser("version", help="Bump version and release")
    version_parser.add_argument(
        "--from", dest="from_version", required=True, help="Current version string"
    )
    version_parser.add_argument(
        "--to", dest="to_version", required=True, help="New version string"
    )
    version_parser.add_argument(
        "--prev", dest="prev_version", default=None,
        help="Previous version for sidebars inference"
    )

    # Spark variant support
    version_parser.add_argument(
        "--spark-variant", dest="spark_variant", default=None,
        choices=["spark3.3", "spark3.4", "spark3.5", "spark4.0"],
        help="Spark variant for release (updates build.sbt, creates variant tag)"
    )
    version_parser.add_argument(
        "--spark-profile", dest="spark_profile_path",
        default=None,
        help="Path to spark_profiles.yaml (default: tools/release/spark_profiles.yaml)"
    )

    # Apply and scope control
    version_parser.add_argument(
        "--apply", action="store_true", help="Apply edits; otherwise dry-run preview"
    )
    version_parser.add_argument(
        "--undo-docs", action="store_true", help="Undo docs versioning for --to version"
    )
    version_parser.add_argument(
        "--root", default=str(Path.cwd()), help="Repository root (default: CWD)"
    )
    version_parser.add_argument(
        "--skip-bump", action="store_true", help="Skip version string bumping"
    )
    version_parser.add_argument(
        "--skip-docs", action="store_true",
        help="Skip docs snapshot and versions.json update"
    )
    version_parser.add_argument(
        "--skip-build-config", action="store_true",
        help="Skip build.sbt, PackageUtils.scala updates (only for --spark-variant)"
    )
    version_parser.add_argument(
        "--skip-docker", action="store_true",
        help="Skip Dockerfile and start script updates"
    )

    # Verification
    version_parser.add_argument(
        "--verify", action="store_true",
        help="Verify results (uses rg if available, else Python)"
    )
    version_parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any old version occurrences remain after apply"
    )

    # Git operations
    version_parser.add_argument("--git-commit", action="store_true", help="Run git add/commit")
    version_parser.add_argument("--git-tag", action="store_true", help="Create annotated git tag")
    version_parser.add_argument("--git-push", action="store_true", help="Push HEAD and tags to remote")
    version_parser.add_argument("--commit-message", default=None, help="Custom commit message")
    version_parser.add_argument("--tag-name", default=None, help="Custom tag name (default v<to>)")
    version_parser.add_argument("--tag-message", default=None, help="Custom tag message")
    version_parser.add_argument("--remote", default="origin", help="Git remote to push to")

    args = ap.parse_args()

    # Handle legacy invocation (no subcommand)
    if args.command is None:
        # Check if --from was passed directly (legacy mode)
        if hasattr(args, 'from_version') and args.from_version:
            args.command = "version"
        else:
            ap.print_help()
            sys.exit(1)

    if args.command == "version":
        run_version_command(args)


def run_version_command(args) -> None:
    """Execute the version bump command."""
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Load Spark profiles
    profile_path = Path(args.spark_profile_path) if args.spark_profile_path else root / "tools" / "release" / "spark_profiles.yaml"
    profiles = load_spark_profiles(profile_path)

    # Get current and target profiles
    current_profile = detect_current_profile(root)
    target_profile = None
    if args.spark_variant:
        if args.spark_variant in profiles:
            target_profile = profiles[args.spark_variant]
        else:
            print(f"WARNING: Spark variant '{args.spark_variant}' not found in profiles")

    # Determine tag name for pre-flight checks
    tag_name = args.tag_name
    if not tag_name and args.git_tag:
        if args.spark_variant and args.spark_variant in profiles:
            suffix = profiles[args.spark_variant].tag_suffix
            tag_name = f"v{args.to_version}{suffix}"
        else:
            tag_name = f"v{args.to_version}"

    # Print header
    print("=" * 60)
    print("SynapseML Release Automation")
    print("=" * 60)
    print(f"  SynapseML Version: {args.from_version} -> {args.to_version}")
    if args.spark_variant:
        print(f"  Spark Variant: {args.spark_variant}")
        if target_profile and current_profile:
            print(f"  Spark Version: {current_profile.spark_version} -> {target_profile.spark_version}")
            print(f"  Scala Version: {current_profile.scala_version} -> {target_profile.scala_version}")
    if tag_name:
        print(f"  Tag Name: {tag_name}")
    print(f"  Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    # Pre-flight checks
    checker = PreflightChecker(root, args.from_version, args.to_version, args.spark_variant)
    passed, errors, warnings = checker.run_all(tag_name if args.git_tag else None)

    if errors and not args.apply:
        print("\nPre-flight errors detected. Review before applying.")
    elif errors and args.apply:
        print("\nERROR: Pre-flight checks failed. Aborting.")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # Phase 1: Version String Updates
    if not args.skip_bump:
        print("\n" + "=" * 60)
        print("Phase 1: Version String Updates")
        print("=" * 60)

        from_re = re.compile(re.escape(args.from_version))
        files = collect_files(root)
        files = [
            f for f in files
            if f.suffix in {".md", ".js", ".ts", ".json", ".yaml"}
            or f.name in {"README.md", "start"}
        ]

        if not args.apply:
            preview_matches(files, from_re, root)
        else:
            count = apply_replacements(files, from_re, args.to_version, root)
            print(f"\n  Version bump complete: {count} files updated")

    # Phase 2: Build Configuration Updates (Spark variants only)
    if args.spark_variant and not args.skip_build_config and current_profile and target_profile:
        print("\n" + "=" * 60)
        print("Phase 2: Build Configuration Updates")
        print("=" * 60)

        all_changes = apply_build_config_updates(
            root, current_profile, target_profile,
            dry_run=not args.apply,
            skip_docker=args.skip_docker
        )

        if all_changes:
            for file_name, changes in all_changes.items():
                prefix = "[DRY-RUN] " if not args.apply else "  Updated: "
                print(f"{prefix}{file_name}")
                for change in changes:
                    print(f"    - {change}")
        else:
            print("  No build configuration changes needed")

    # Phase 3: Documentation Versioning
    if args.undo_docs:
        print("\n" + "=" * 60)
        print("Undo Documentation Versioning")
        print("=" * 60)
        website_dir = root / "website"
        if website_dir.is_dir():
            undo_docs_version(website_dir, args.to_version)
        else:
            print("  Skipped: missing website/ directory")

    elif not args.skip_docs:
        print("\n" + "=" * 60)
        print("Phase 3: Documentation Versioning")
        print("=" * 60)

        # For Spark variants, typically skip docs versioning (use base version docs)
        if args.spark_variant:
            print("  Spark variant release: docs versioning skipped (uses base version docs)")
        elif args.apply:
            website_dir = root / "website"
            docs_src = root / "docs"
            dest_dir = website_dir / "versioned_docs" / f"version-{args.to_version}"
            sidebars_dir = website_dir / "versioned_sidebars"
            versions_json = website_dir / "versions.json"

            if not website_dir.is_dir() or not docs_src.is_dir():
                print("  Skipped: missing website/ or docs/ directory")
            else:
                used_docusaurus = run_docusaurus_docs_version(website_dir, args.to_version)
                if not used_docusaurus:
                    prev = args.prev_version or infer_prev_version(versions_json)
                    copy_docs_offline(docs_src, dest_dir)
                    print(f"  Created versioned docs at {dest_dir.relative_to(root).as_posix()}")
                    copy_sidebars_offline(sidebars_dir, prev, args.to_version)
                    update_versions_json(versions_json, args.to_version)
                print("  Docs versioning complete")
        else:
            website_dir = root / "website"
            dest_dir = website_dir / "versioned_docs" / f"version-{args.to_version}"
            sidebars_json = website_dir / "versioned_sidebars" / f"version-{args.to_version}-sidebars.json"
            print("  [DRY-RUN] Would create:")
            print(f"    - {dest_dir.relative_to(root).as_posix()}")
            print(f"    - {sidebars_json.relative_to(root).as_posix()}")
            print(f"    - Update versions.json")

    # Phase 4: Verification
    if args.verify:
        print("\n" + "=" * 60)
        print("Phase 4: Verification")
        print("=" * 60)

        if which("rg"):
            verify_with_rg(root, args.from_version, args.to_version)
        else:
            verify_python(root, args.from_version, args.to_version)

    # Phase 5: Git Operations
    if args.git_commit or args.git_tag or args.git_push:
        if not args.apply:
            print("\n" + "=" * 60)
            print("Git Operations (skipped in dry-run)")
            print("=" * 60)
            print(f"  Would commit, tag as {tag_name}, and push" if args.git_push else f"  Would commit and tag as {tag_name}")
        else:
            run_git_steps(
                root,
                args.to_version,
                args.spark_variant,
                profiles,
                args.git_commit,
                args.git_tag,
                args.git_push,
                args.commit_message,
                args.tag_name,
                args.tag_message,
                args.remote,
            )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if args.apply:
        print(f"  Release {args.to_version} preparation complete")
        if args.spark_variant:
            print(f"  Spark variant: {args.spark_variant}")
        if args.git_tag:
            print(f"  Tag: {tag_name}")
    else:
        print("  DRY-RUN complete. Use --apply to execute changes.")
        print(f"\n  To apply: python tools/release/release.py version --from {args.from_version} --to {args.to_version}" +
              (f" --spark-variant {args.spark_variant}" if args.spark_variant else "") +
              " --apply")


if __name__ == "__main__":
    main()
