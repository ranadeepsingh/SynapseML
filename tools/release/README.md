# SynapseML Release Helper

This folder contains a single end-to-end script that safely bumps version strings across the repo, versions documentation, and optionally verifies and commits/tag/pushes changes.

## Script

- release.py
  - One-call release helper that:
    - Bumps version strings in targeted text files.
    - Versions docs into `website/versioned_docs/version-<to>` and updates sidebars and `website/versions.json` (prefers Docusaurus, falls back to offline copy).
    - Optionally verifies changes and runs git add/commit/tag/push.
  - Usage:

    ```bash
    # Dry-run preview of changes (no writes)
    python tools/release/release.py --from 1.0.15 --to 1.0.16

    # Apply bump + docs snapshot, then commit, tag, and push
    python tools/release/release.py --from 1.0.15 --to 1.0.16 \
      --prev 1.0.15 --apply --verify --git-commit --git-tag --git-push

    # Undo docs versioning for a specific version
    python tools/release/release.py --from 1.0.15 --to 1.0.16 --undo-docs
    ```

  - Flags:
    - `--from`, `--to`: version strings
    - `--prev`: previous version (used to copy sidebars if Docusaurus isn’t available; inferred from `website/versions.json` if omitted)
    - `--apply`: perform edits; omit for dry-run
    - `--verify`: check changes (uses `rg` if available, else Python)
    - `--git-commit`, `--git-tag`, `--git-push`: optional git steps; customize with `--commit-message`, `--tag-name`, `--tag-message`, `--remote`
    - `--skip-bump`, `--skip-docs`: skip specific phases

## Typical Flow (1.0.15 → 1.0.16)

```bash
python tools/release/release.py --from 1.0.15 --to 1.0.16 \
  --prev 1.0.15 --apply --verify --git-commit --git-tag --git-push
```

## How files are selected

- Includes: `README.md`, `start`, `docs/**`, `website/**` (non-versioned), `tools/docker/**`, and text file types `.md`, `.js`, `.ts`, `.json`, `.yaml`.
- Excludes: `website/versioned_docs/**`, `website/versions.json`, `tools/release/**`, `node_modules/**`, `.git/**`, lockfiles and images.
- Respects .gitignore: ignored files are automatically skipped using `git check-ignore`; if git is unavailable, a best-effort parse of `.gitignore` patterns is used.

## Docusaurus notes

- Prefers `docusaurus docs:version <to>` via yarn or npx. If unavailable or it fails, falls back to copying `docs/` to `website/versioned_docs/version-<to>`, copying sidebars, and updating `versions.json`.
- After versioning docs, you can run `yarn build` in `website/` and deploy as needed.

## Safety rules

- The bump step avoids editing frozen docs and only targets text files.
- Always run a dry-run first to confirm the file list and snippets.

## Artifact & links checklist

- Ensure `com.microsoft.azure:synapseml_2.12:<ver>` is published.
- Ensure `synapseml==<ver>` (PyPI) and Dotnet packages (if any) are published.
- Ensure blob docs and example DBC exist:
  - `https://mmlspark.blob.core.windows.net/docs/<ver>/*`
  - `https://mmlspark.blob.core.windows.net/dbcs/SynapseMLExamplesv<ver>.dbc`
