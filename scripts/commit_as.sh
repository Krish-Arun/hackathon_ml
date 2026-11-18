#!/usr/bin/env bash
# Commit as a specific team member from a shared laptop.
#
#   ./scripts/commit_as.sh krish "feat: candidate generation"
#   ./scripts/commit_as.sh niharika "feat: viterbi decoder"
#   ./scripts/commit_as.sh mrudul "feat: fastapi /predict endpoint"
#
# Sets BOTH author and committer so `git log` and the GitHub UI agree.
# The push credential is separate and stays whoever's laptop this is.
set -euo pipefail

case "${1:-}" in
  krish)    NAME="Krish Arun";     EMAIL="krish.official.v1@gmail.com" ;;
  niharika) NAME="Niharika Mirle"; EMAIL="neha.mirle17@gmail.com" ;;
  mrudul)   NAME="Mrudul";         EMAIL="chinju1206@gmail.com" ;;
  *) echo "usage: $0 {krish|niharika|mrudul} \"commit message\"" >&2; exit 1 ;;
esac

shift
MSG="${*:-}"
[ -z "$MSG" ] && { echo "error: empty commit message" >&2; exit 1; }

GIT_AUTHOR_NAME="$NAME"    GIT_AUTHOR_EMAIL="$EMAIL" \
GIT_COMMITTER_NAME="$NAME" GIT_COMMITTER_EMAIL="$EMAIL" \
  git commit -m "$MSG"

echo "committed as $NAME <$EMAIL>"
