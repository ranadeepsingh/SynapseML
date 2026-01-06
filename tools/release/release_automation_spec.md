# SynapseML Release Automation Specification

Version: 1.0
Last Updated: December 2025

## Overview

This specification defines the release automation system for SynapseML, enabling:
- **SynapseML version bumps** (e.g., 1.1.0 → 1.1.1)
- **Spark-variant releases** (e.g., v1.1.1-spark4.0 with Spark 4.0.1, Scala 2.13)
- **Safe, automated** version string updates across all relevant files
- **Build configuration updates** for different Spark versions

---

## Architecture

```
tools/release/
├── release.py              # Main CLI tool
├── spark_profiles.yaml     # Spark variant configurations
├── release_automation_spec.md  # This specification
└── README.md               # User documentation
```

---

## CLI Interface

### Command Structure

```bash
python tools/release/release.py version \
  --from <current_version> \
  --to <new_version> \
  [--spark-variant <variant>] \
  [--apply] [--verify] \
  [--git-commit] [--git-tag] [--git-push]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--from` | Yes | Current version string (e.g., `1.1.0`) |
| `--to` | Yes | New version string (e.g., `1.1.1`) |
| `--spark-variant` | No | Spark variant: `spark3.3`, `spark3.4`, `spark3.5`, `spark4.0` |
| `--spark-profile` | No | Path to custom spark_profiles.yaml |
| `--apply` | No | Apply changes (default: dry-run) |
| `--verify` | No | Verify changes after apply |
| `--strict` | No | Fail if any old version occurrences remain |
| `--skip-bump` | No | Skip version string bumping |
| `--skip-docs` | No | Skip documentation versioning |
| `--skip-build-config` | No | Skip build.sbt, PackageUtils.scala updates |
| `--skip-docker` | No | Skip Dockerfile and start script updates |
| `--git-commit` | No | Stage and commit changes |
| `--git-tag` | No | Create annotated git tag |
| `--git-push` | No | Push to remote (default: origin) |
| `--remote` | No | Git remote name (default: `origin`) |
| `--undo-docs` | No | Undo documentation versioning for `--to` version |

---

## Spark Profiles Configuration

### File: `spark_profiles.yaml`

```yaml
# Default profile (master branch, Spark 3.5)
default:
  spark_version: "3.5.0"
  scala_version: "2.12.17"
  scala_major: "2.12"
  java_version: "11"
  python_version: "3.11.8"
  pyspark_version: "3.5.0"
  pyarrow_version: "10.0.1"
  hadoop_version: "3"
  isolation_forest_spark: "3.5.0"
  isolation_forest_version: "3.0.5"
  branch: "master"
  tag_suffix: ""

spark3.3:
  spark_version: "3.3.2"
  scala_version: "2.12.15"
  scala_major: "2.12"
  java_version: "11"
  python_version: "3.10"
  pyspark_version: "3.3.2"
  pyarrow_version: "10.0.1"
  hadoop_version: "3"
  isolation_forest_spark: "3.3.0"
  isolation_forest_version: "3.0.5"
  branch: "spark3.3"
  tag_suffix: "-spark3.3"

spark3.4:
  spark_version: "3.4.1"
  scala_version: "2.12.17"
  scala_major: "2.12"
  java_version: "11"
  python_version: "3.10"
  pyspark_version: "3.4.1"
  pyarrow_version: "10.0.1"
  hadoop_version: "3"
  isolation_forest_spark: "3.4.0"
  isolation_forest_version: "3.0.5"
  branch: "spark3.4"
  tag_suffix: "-spark3.4"

spark3.5:
  spark_version: "3.5.0"
  scala_version: "2.12.17"
  scala_major: "2.12"
  java_version: "11"
  python_version: "3.11.8"
  pyspark_version: "3.5.0"
  pyarrow_version: "10.0.1"
  hadoop_version: "3"
  isolation_forest_spark: "3.5.0"
  isolation_forest_version: "3.0.5"
  branch: "master"
  tag_suffix: ""  # spark3.5 is default, no suffix

spark4.0:
  spark_version: "4.0.1"
  scala_version: "2.13.16"
  scala_major: "2.13"
  java_version: "17"
  python_version: "3.12.11"
  pyspark_version: "4.0.1"
  pyarrow_version: "22.0.0"
  hadoop_version: "3"
  isolation_forest_spark: "4.0.1"
  isolation_forest_version: "4.0.7"
  branch: "spark4.0"
  tag_suffix: "-spark4.0"
```

---

## File Update Rules

### Category 1: SynapseML Version Only (All Releases)

These files are updated for ALL releases (both standard and Spark variants):

| File | Line(s) | Pattern | Notes |
|------|---------|---------|-------|
| `README.md` | 14, 17, 129, 145, 181, 189 | `1.1.0` literal | Badges, install instructions |
| `website/docusaurus.config.js` | 4, 16 | `version = "1.1.0"` | Website version constant |
| `docs/**/*.md` | Various | `synapseml_2.12:1.1.0` | Installation examples |
| `website/src/pages/index.js` | Various | Version in code snippets | Homepage examples |
| `tools/docker/demo/Dockerfile` | 3 | `SYNAPSEML_VERSION=1.1.0` | Docker build arg |
| `tools/docker/minimal/Dockerfile` | 3 | `SYNAPSEML_VERSION=1.1.0` | Docker build arg |
| `start` | 7 | `SYNAPSEML_VERSION="1.1.0"` | Binder script |

### Category 2: Spark/Scala Version (Spark Variants Only)

These files are ONLY updated when `--spark-variant` is specified:

| File | Line(s) | Pattern | Example Change |
|------|---------|---------|----------------|
| `build.sbt` | 10 | `val sparkVersion = "X.Y.Z"` | `"3.5.0"` → `"4.0.1"` |
| `build.sbt` | 13 | `scalaVersion := "X.Y.Z"` | `"2.12.17"` → `"2.13.16"` |
| `build.sbt` | 15 | `val scalaMajorVersion = X.YZ` | `2.12` → `2.13` |
| `build.sbt` | 38 | `isolation-forest_X.Y.Z` | `3.5.0` → `4.0.1` |
| `core/.../PackageUtils.scala` | 24 | `spark-avro_2.XX:X.Y.Z` | `2.12:3.5.0` → `2.13:4.0.1` |
| `environment.yml` | Various | `pyspark==X.Y.Z` | Version updates |
| `tools/docker/demo/Dockerfile` | 6 | `SPARK_VERSION=X.Y.Z` | `3.5.0` → `4.0.1` |
| `tools/docker/minimal/Dockerfile` | 6 | `SPARK_VERSION=X.Y.Z` | Same |
| `start` | 5 | `SPARK_VERSION="X.Y.Z"` | Same |

### Category 3: Excluded Files (Never Modify)

```
website/versioned_docs/**     # Frozen documentation snapshots
website/versions.json         # Handled by Docusaurus versioning
tools/release/**              # Release tooling itself
node_modules/**               # Dependencies
.git/**                       # Git internals
target/**                     # Build outputs
**/*.lock                     # Lock files
**/*.svg, *.png, *.jpg, etc.  # Image files
```

---

## Regex Patterns

### SynapseML Version
```python
# Simple literal replacement (escaped)
pattern = re.escape(from_version)  # e.g., "1\.1\.0"
replacement = to_version           # e.g., "1.1.1"
```

### build.sbt Patterns
```python
# sparkVersion
r'(val sparkVersion = ")[^"]+(")'
# Replacement: rf'\g<1>{new_spark_version}\g<2>'

# scalaVersion
r'(scalaVersion := ")[^"]+(")'
# Replacement: rf'\g<1>{new_scala_version}\g<2>'

# scalaMajorVersion
r'(val scalaMajorVersion = )[0-9.]+'
# Replacement: rf'\g<1>{new_scala_major}'

# isolation-forest dependency
r'(isolation-forest_)[0-9.]+'
# Replacement: rf'\g<1>{new_spark_version}'
```

### PackageUtils.scala Pattern
```python
# spark-avro coordinate
r'(spark-avro_)([0-9.]+):([0-9.]+)'
# Groups: (1) prefix, (2) scala_major, (3) spark_version
# Replacement: rf'\g<1>{new_scala_major}:{new_spark_version}'
```

### Dockerfile/Start Script Patterns
```python
# SPARK_VERSION in Dockerfile
r'(SPARK_VERSION=)[0-9.]+'
# Replacement: rf'\g<1>{new_spark_version}'

# SPARK_VERSION in start script
r'(SPARK_VERSION=")[^"]+(")'
# Replacement: rf'\g<1>{new_spark_version}\g<2>'
```

---

## Safety Mechanisms

### 1. Pre-flight Checks

Before applying any changes, the tool validates:

| Check | Requirement | Action on Failure |
|-------|-------------|-------------------|
| Git clean | No uncommitted changes | Error, abort |
| Version exists | `--from` version found in README.md | Error, abort |
| Tag not exists | `v{to_version}` tag doesn't exist | Error, abort |
| Branch match | Current branch matches profile.branch | Warning |
| Profile match | build.sbt versions match profile | Warning |

### 2. Dry-Run Mode (Default)

When `--apply` is NOT specified:
- Shows all files that would be modified
- Shows sample context of changes (first 3 matches per file)
- Shows build config changes (for Spark variants)
- No files are modified

### 3. Verification (`--verify`)

After applying changes:
- Scans all target files for remaining old version occurrences
- Reports count of new version occurrences
- With `--strict`: fails if any old version found

### 4. Backup/Restore

- Creates timestamped backup of all modified files before changes
- Backup location: system temp directory
- Manifest file tracks all backed-up files
- Restore function available for rollback

---

## Tag Naming Convention

| Release Type | Tag Format | Example | Branch |
|--------------|------------|---------|--------|
| Standard (Spark 3.5) | `v{version}` | `v1.1.1` | master |
| Spark 4.0 variant | `v{version}-spark4.0` | `v1.1.1-spark4.0` | spark4.0 |
| Spark 3.4 variant | `v{version}-spark3.4` | `v1.1.1-spark3.4` | spark3.4 |
| Spark 3.3 variant | `v{version}-spark3.3` | `v1.1.1-spark3.3` | spark3.3 |

---

## Usage Examples

### Example 1: Standard Release (1.1.0 → 1.1.1)

```bash
# Ensure on master branch
git checkout master
git pull origin master

# Preview changes (dry-run)
python tools/release/release.py version --from 1.1.0 --to 1.1.1

# Apply changes, verify, and create release
python tools/release/release.py version \
  --from 1.1.0 --to 1.1.1 \
  --apply \
  --verify \
  --git-commit \
  --git-tag \
  --git-push

# Result: Creates tag v1.1.1 on master
```

### Example 2: Spark 4.0 Variant Release

```bash
# Switch to spark4.0 branch
git checkout spark4.0
git pull origin spark4.0

# Preview ALL changes (version strings + build config)
python tools/release/release.py version \
  --from 1.1.0 --to 1.1.1 \
  --spark-variant spark4.0

# Apply and release
python tools/release/release.py version \
  --from 1.1.0 --to 1.1.1 \
  --spark-variant spark4.0 \
  --apply \
  --verify \
  --git-commit \
  --git-tag \
  --git-push

# Result: Creates tag v1.1.1-spark4.0 on spark4.0 branch
```

### Example 3: Version Bump Only (No Docs)

```bash
python tools/release/release.py version \
  --from 1.1.0 --to 1.1.1 \
  --skip-docs \
  --apply
```

### Example 4: Undo Failed Release

```bash
# Undo documentation versioning
python tools/release/release.py version \
  --from 1.1.0 --to 1.1.1 \
  --undo-docs

# Revert file changes
git checkout -- .
```

---

## Implementation Notes

### Dependencies

- Python 3.11+ (standard library only)
- Optional: PyYAML for spark_profiles.yaml (fallback to JSON if not available)
- Optional: ripgrep (`rg`) for faster verification

### Error Handling

- All file operations wrapped in try/except
- Graceful handling of missing files
- Clear error messages with file paths
- Non-zero exit code on failure

### Logging

- Verbose output for all operations
- Clear phase markers (Pre-flight, Version Bump, Build Config, etc.)
- Summary at end showing files modified

---

## Appendix: File Locations

### Critical Files for Version Updates

```
# SynapseML Version References
README.md
website/docusaurus.config.js
website/src/pages/index.js
docs/Get Started/Install SynapseML.md
docs/Get Started/Quickstart - Your First Models.md
tools/docker/demo/Dockerfile
tools/docker/minimal/Dockerfile
start

# Build Configuration (Spark Variants)
build.sbt
core/src/main/scala/com/microsoft/azure/synapse/ml/core/env/PackageUtils.scala
environment.yml

# Documentation Versioning
website/versions.json
website/versioned_docs/
website/versioned_sidebars/
```

### Git Branches

```
master          # Default Spark 3.5 releases
spark3.3        # Spark 3.3 maintenance
spark3.4        # Spark 3.4 maintenance
spark3.5        # Spark 3.5 (mirrors master)
spark4.0        # Spark 4.0 releases
```

---

## Changelog

- **v1.0** (December 2025): Initial specification
  - Support for SynapseML version bumps
  - Support for Spark variant releases
  - Spark profiles configuration
  - Safety mechanisms and verification
