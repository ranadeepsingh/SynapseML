# SynapseML Release Automation

Automated release tooling for SynapseML version bumps and Spark variant releases.

## Quick Start

```bash
# Preview changes (dry-run)
python tools/release/release.py version --from 1.1.0 --to 1.1.1

# Apply standard release (Spark 3.5)
python tools/release/release.py version --from 1.1.0 --to 1.1.1 \
    --apply --verify --git-commit --git-tag --git-push

# Apply Spark 4.0 variant release
python tools/release/release.py version --from 1.1.0 --to 1.1.1 \
    --spark-variant spark4.0 --apply --git-commit --git-tag --git-push
```

## Features

- **Version String Updates**: Automatically updates version references across README, docs, website, and Docker files
- **Spark Variant Support**: Creates releases for different Spark versions (3.3, 3.4, 3.5, 4.0) with appropriate build.sbt changes
- **Documentation Versioning**: Integrates with Docusaurus to create versioned documentation snapshots
- **Safety Mechanisms**: Pre-flight checks, dry-run mode, and verification
- **Git Integration**: Automated commit, tag, and push operations

## Usage

### Standard Release

For a standard release on the master branch (Spark 3.5):

```bash
# Step 1: Preview changes
python tools/release/release.py version --from 1.1.0 --to 1.1.1

# Step 2: Apply changes
python tools/release/release.py version --from 1.1.0 --to 1.1.1 --apply

# Step 3: Verify, commit, tag, and push
python tools/release/release.py version --from 1.1.0 --to 1.1.1 \
    --apply --verify --git-commit --git-tag --git-push
```

This creates tag `v1.1.1`.

### Spark Variant Release

For a Spark 4.0 variant release:

```bash
# Switch to spark4.0 branch first
git checkout spark4.0
git pull origin spark4.0

# Preview changes (includes build.sbt updates)
python tools/release/release.py version --from 1.1.0 --to 1.1.1 --spark-variant spark4.0

# Apply and release
python tools/release/release.py version --from 1.1.0 --to 1.1.1 \
    --spark-variant spark4.0 --apply --verify --git-commit --git-tag --git-push
```

This creates tag `v1.1.1-spark4.0`.

## Command Reference

### Arguments

| Argument | Description |
|----------|-------------|
| `--from VERSION` | Current version string (required) |
| `--to VERSION` | New version string (required) |
| `--spark-variant VARIANT` | Spark variant: `spark3.3`, `spark3.4`, `spark3.5`, `spark4.0` |
| `--apply` | Apply changes (default: dry-run preview) |
| `--verify` | Verify changes after apply |
| `--strict` | Fail if any old version occurrences remain |
| `--skip-bump` | Skip version string updates |
| `--skip-docs` | Skip documentation versioning |
| `--skip-build-config` | Skip build.sbt and PackageUtils.scala updates |
| `--skip-docker` | Skip Dockerfile and start script updates |
| `--git-commit` | Stage and commit changes |
| `--git-tag` | Create annotated git tag |
| `--git-push` | Push to remote |
| `--undo-docs` | Undo documentation versioning for `--to` version |

### Git Arguments

| Argument | Description |
|----------|-------------|
| `--commit-message MSG` | Custom commit message |
| `--tag-name NAME` | Custom tag name (default: `v{version}`) |
| `--tag-message MSG` | Custom tag message |
| `--remote NAME` | Git remote name (default: `origin`) |

## Files Updated

### All Releases

| File | Updates |
|------|---------|
| `README.md` | Version badges, installation instructions |
| `website/docusaurus.config.js` | Version constant |
| `docs/**/*.md` | Version strings in examples |
| `tools/docker/*/Dockerfile` | `SYNAPSEML_VERSION` |
| `start` | `SYNAPSEML_VERSION` |

### Spark Variant Releases (additional)

| File | Updates |
|------|---------|
| `build.sbt` | `sparkVersion`, `scalaVersion`, `scalaMajorVersion` |
| `core/.../PackageUtils.scala` | `spark-avro` coordinate |
| `environment.yml` | `pyspark`, `python` versions |
| `tools/docker/*/Dockerfile` | `SPARK_VERSION` |
| `start` | `SPARK_VERSION` |

## Spark Profiles

Spark variant configurations are defined in `spark_profiles.yaml`:

```yaml
spark4.0:
  spark_version: "4.0.1"
  scala_version: "2.13.16"
  scala_major: "2.13"
  java_version: "17"
  pyspark_version: "4.0.1"
  branch: "spark4.0"
  tag_suffix: "-spark4.0"
```

## Tag Naming Convention

| Release Type | Tag Format | Example |
|--------------|------------|---------|
| Standard (Spark 3.5) | `v{version}` | `v1.1.1` |
| Spark 4.0 | `v{version}-spark4.0` | `v1.1.1-spark4.0` |
| Spark 3.4 | `v{version}-spark3.4` | `v1.1.1-spark3.4` |
| Spark 3.3 | `v{version}-spark3.3` | `v1.1.1-spark3.3` |

## Troubleshooting

### Undo a Failed Release

If something goes wrong during a release:

```bash
# Undo documentation versioning
python tools/release/release.py version --from 1.1.0 --to 1.1.1 --undo-docs

# Revert file changes
git checkout -- .

# Delete local tag (if created)
git tag -d v1.1.1
```

### Pre-flight Check Failures

- **"Version X not found in README.md"**: Ensure `--from` matches the current version
- **"Tag already exists"**: Use a different version or delete the existing tag
- **"Git working directory has uncommitted changes"**: Commit or stash changes first

## Requirements

- Python 3.11+
- Git
- Optional: PyYAML (for profile loading)
- Optional: ripgrep (`rg`) for faster verification

## Files

```
tools/release/
├── release.py               # Main CLI tool
├── spark_profiles.yaml      # Spark variant configurations
├── release_automation_spec.md  # Full specification
└── README.md                # This file
```
