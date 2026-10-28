#!/usr/bin/env python3
"""Work around a missing registry row for one Terminal-Bench 2.1 task.

The task_version row for one task (gpt2-codegolf) became invisible upstream, so
the registry returns task_version=None for it. harbor then dereferences that
None while resolving the dataset and dies with a TypeError, which makes every
evaluation on this dataset version impossible over a single missing task.

The task itself is still in the local package cache, under a digest that matches
what earlier trials recorded. So two RegistryDB methods are wrapped to fill the
registry response in from local information and keep the full task set:

  - get_dataset_version_tasks: fills rows whose task_version is None with the
    known (org, name, digest). Fails loudly on any unknown missing row.
  - resolve_task_version: short-circuits resolution of the known digest, after
    checking that the local cache actually holds it.

The patch is appended to the installed harbor package behind a marker:

  - idempotent: a no-op if the marker is already present
  - reinstalling harbor (for instance via `uv sync`) removes it, so
    scripts/run_eval.sh re-applies it on every run
  - if harbor changes such that the wrapped methods no longer exist, it stops
    with an error rather than silently doing nothing

The upstream row was restored at some point, in which case all of this is inert:
with no None rows the first wrapper returns unchanged, and without a matching
digest and cache entry the second defers to the original. It is kept as
insurance against the same class of outage.

Run directly with:
    uv run python scripts/apply_harbor_registry_fix.py
"""

import sys
from pathlib import Path

MARKER = "# --- TB2.1 gpt2-codegolf registry workaround (scripts/apply_harbor_registry_fix.py) ---"

PATCH = f'''

{MARKER}
def _tb21_apply_gpt2codegolf_fix():
    import functools

    _MISSING = {{
        ("terminal-bench", "gpt2-codegolf"):
            "fe42af8e9c5aa927c3b680acd08e43032c84088fa9d770b03ce63dbc66fea4e7",
    }}

    def _cache_dir(org, name, digest):
        from harbor.constants import PACKAGE_CACHE_DIR
        return PACKAGE_CACHE_DIR / org / name / digest

    _orig_tasks = RegistryDB.get_dataset_version_tasks

    @functools.wraps(_orig_tasks)
    async def get_dataset_version_tasks(self, dataset_version_id):
        rows = await _orig_tasks(self, dataset_version_id)
        if all(r.get("task_version") for r in rows):
            return rows
        fixed = [r for r in rows if r.get("task_version")]
        visible = {{
            r["task_version"]["package"]["name"]
            for r in fixed
            if r["task_version"].get("package")
        }}
        for (org, name), digest in _MISSING.items():
            if name in visible:
                continue
            if not _cache_dir(org, name, digest).is_dir():
                raise RuntimeError(
                    f"registry row for {{org}}/{{name}} is hidden and local task cache "
                    f"{{_cache_dir(org, name, digest)}} is missing: cannot repair dataset"
                )
            fixed.append({{
                "task_version": {{
                    "content_hash": digest,
                    "package": {{"name": name, "org": {{"name": org}}}},
                }}
            }})
        if len(fixed) != len(rows):
            raise RuntimeError(
                f"{{len(rows) - len(fixed)}} registry row(s) with task_version=None beyond "
                f"the known gpt2-codegolf workaround — inspect dataset "
                f"{{dataset_version_id}} before proceeding"
            )
        return fixed

    RegistryDB.get_dataset_version_tasks = get_dataset_version_tasks

    _orig_resolve = RegistryDB.resolve_task_version

    @functools.wraps(_orig_resolve)
    async def resolve_task_version(self, org, name, ref="latest"):
        digest = _MISSING.get((org, name))
        if digest and ref in (f"sha256:{{digest}}", digest) \\
                and _cache_dir(org, name, digest).is_dir():
            # archive_path and id are unused on a cache hit
            return ResolvedTaskVersion(
                id="00000000-0000-0000-0000-000000000000",
                archive_path="",
                content_hash=digest,
            )
        return await _orig_resolve(self, org, name, ref)

    RegistryDB.resolve_task_version = resolve_task_version


_tb21_apply_gpt2codegolf_fix()
del _tb21_apply_gpt2codegolf_fix
'''


def main() -> int:
    try:
        import harbor.db.client as mod
    except ImportError:
        print("apply_harbor_registry_fix: cannot import harbor (outside the venv?)", file=sys.stderr)
        return 1

    target = Path(mod.__file__)
    text = target.read_text()

    if MARKER in text:
        print(f"apply_harbor_registry_fix: already applied ({target})")
        return 0

    # Check the premise: the wrapped methods must exist in this harbor version
    for needle in (
        "class RegistryDB",
        "class ResolvedTaskVersion",
        "async def get_dataset_version_tasks",
        "async def resolve_task_version",
    ):
        if needle not in text:
            print(
                f"apply_harbor_registry_fix: '{needle}' not found in {target}.\n"
                "  harbor has changed; revisit the patch before using it.",
                file=sys.stderr,
            )
            return 1

    target.write_text(text + PATCH)
    print(f"apply_harbor_registry_fix: applied -> {target}")

    # Verify immediately that the appended block imports cleanly
    import importlib
    importlib.reload(mod)
    return 0


if __name__ == "__main__":
    sys.exit(main())
