#!/usr/bin/env bash
# Regenerates assets/css/chroma.css (syntax highlighting, light + dark).
# The output is checked in so builds have no generation step.
set -euo pipefail

cd "$(dirname "$0")/.."

LIGHT=github
DARK=github-dark

{
  echo "/* Generated: hugo gen chromastyles. Regenerate with scripts/gen-chroma.sh */"
  hugo gen chromastyles --style="$LIGHT"
  echo ""
  echo "@media (prefers-color-scheme: dark) {"
  hugo gen chromastyles --style="$DARK" | sed 's/^/  /'
  echo "}"
} > assets/css/chroma.css

echo "wrote assets/css/chroma.css ($LIGHT / $DARK)"
