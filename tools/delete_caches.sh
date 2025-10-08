#!/usr/bin/env bash
# delete_caches.sh
# Delete .pkl and .lmdb in the splits/ folder (train/test) ONLY.
# Does NOT touch any .pkl under datas/.

set -euo pipefail

usage() {
  echo "Usage: $0 /path/to/dataset_root [--dry-run]"
  echo "  dataset_root should contain splits/{train,test}"
  exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
fi

DATASET_ROOT="$1"
DRY_RUN="${2:-}"

SPLIT_DIR="${DATASET_ROOT%/}/splits"
if [[ ! -d "$SPLIT_DIR" ]]; then
  echo "Error: '$SPLIT_DIR' not found. Expected 'splits/' under the dataset root."
  exit 2
fi

# Pre-initialize to avoid unbound array with `set -u`
FILES=()
DIRS=()

# Collect target files and directories inside splits/
while IFS= read -r -d '' p; do FILES+=("$p"); done < <(
  find "$SPLIT_DIR" -type f \( -name '*.pkl' -o -name '*.lmdb' \) -print0
)
while IFS= read -r -d '' p; do DIRS+=("$p"); done < <(
  find "$SPLIT_DIR" -type d -name '*.lmdb' -print0
)

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "DRY RUN — nothing will be deleted."
  echo "Files that would be removed:"
  ((${#FILES[@]})) && printf '%s\n' "${FILES[@]}" || echo "(none)"
  echo
  echo "Directories that would be removed (recursively):"
  ((${#DIRS[@]})) && printf '%s\n' "${DIRS[@]}" || echo "(none)"
  exit 0
fi

# Delete files and directories (guarded so xargs isn't called on empty input)
((${#FILES[@]})) && printf '%s\0' "${FILES[@]}" | xargs -0 rm -f
((${#DIRS[@]}))  && printf '%s\0' "${DIRS[@]}"  | xargs -0 rm -rf

echo "Done. Removed ${#FILES[@]} files and ${#DIRS[@]} directories under '$SPLIT_DIR'."