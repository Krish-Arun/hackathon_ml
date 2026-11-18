#!/usr/bin/env bash
# Commit as Krish (the person operating this machine), optionally crediting
# teammates who shaped the work as co-authors.
#
#   ./scripts/commit.sh "feat: feature engineering"
#   ./scripts/commit.sh "feat: viterbi decoder" niharika
#   ./scripts/commit.sh "feat: api wiring" niharika mrudul
#
# Only list someone who actually contributed to THAT commit - the trailers are
# a record of who shaped the work, so they should be accurate per commit
# rather than blanket-applied.
#
# For work a teammate wrote themselves, use commit_as.sh instead so they are
# the author proper.
set -euo pipefail

AUTHOR_NAME="Krish Arun"
AUTHOR_EMAIL="krish.official.v1@gmail.com"

MSG="${1:-}"
[ -z "$MSG" ] && { echo "usage: $0 \"message\" [niharika] [mrudul]" >&2; exit 1; }
shift

TRAILERS=""
for who in "$@"; do
  case "$who" in
    niharika) TRAILERS+=$'\n'"Co-authored-by: Niharika Mirle <neha.mirle17@gmail.com>" ;;
    mrudul)   TRAILERS+=$'\n'"Co-authored-by: Mrudul <chinju1206@gmail.com>" ;;
    krish)    ;;  # already the author
    *) echo "unknown teammate: $who (expected niharika or mrudul)" >&2; exit 1 ;;
  esac
done

# Co-authored-by trailers must be in the final paragraph, after a blank line.
FULL="$MSG"
[ -n "$TRAILERS" ] && FULL="$MSG"$'\n'"$TRAILERS"

GIT_AUTHOR_NAME="$AUTHOR_NAME"    GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
GIT_COMMITTER_NAME="$AUTHOR_NAME" GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
  git commit -q -m "$FULL"

echo "committed as $AUTHOR_NAME${TRAILERS:+ (+co-authors)}"
