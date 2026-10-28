#!/usr/bin/env bash
# Install the Claude Code CLI version this setting was run with, locally.
#
# The proposer runs through the Claude Code CLI and its behaviour depends on the
# version: between 2.1.56 and 2.1.220 the output token count roughly doubled,
# and the `opus` model alias started resolving to a different model. The `claude`
# on PATH updates on its own, so reproducing the reported runs means pinning it
# here.
#
# The version is read from CLAUDE_CODE_VERSION in meta_harness.py, which is the
# single source of truth.
#
#   bash scripts/install_claude_cli.sh
#
# Installs into <setting>/.claude-cli/, which is not tracked by git. After that
# the code picks it up automatically: claude_wrapper.CLAUDE_CLI prefers this path
# over PATH. To use an installation elsewhere, set CLAUDE_CLI=/path/to/claude.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ARM_DIR="$PWD"

VERSION="$(python3 - <<'PY'
import ast
for node in ast.parse(open("meta_harness.py").read()).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CLAUDE_CODE_VERSION":
        print(node.value.value); break
else:
    raise SystemExit("ERROR: CLAUDE_CODE_VERSION not found in meta_harness.py")
PY
)"

PREFIX="$ARM_DIR/.claude-cli"
BIN="$PREFIX/node_modules/.bin/claude"

if [[ -x "$BIN" ]] && [[ "$("$BIN" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')" == "$VERSION" ]]; then
    echo "already installed: $BIN ($VERSION)"
    exit 0
fi

command -v npm >/dev/null || { echo "ERROR: npm not found" >&2; exit 1; }

echo "installing Claude Code $VERSION into $PREFIX"
mkdir -p "$PREFIX"
( cd "$PREFIX" && npm install --no-save --silent "@anthropic-ai/claude-code@$VERSION" )

[[ -x "$BIN" ]] || { echo "ERROR: $BIN was not created" >&2; exit 1; }
got="$("$BIN" --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
[[ "$got" == "$VERSION" ]] || { echo "ERROR: version mismatch (got $got, want $VERSION)" >&2; exit 1; }

echo "OK: $BIN ($got)"
echo "meta_harness.py will use this automatically."
