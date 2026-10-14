#!/usr/bin/env bash
# find_gitkeep_and_download.sh
#
# Searches the current directory recursively for .gitkeep files located under
#   **/models/<username>/<reponame>/...
#   **/datasets/<username>/<reponame>/...
# and downloads the contents of the folder containing each .gitkeep from the
# Hugging Face mirror via hfd.sh.
#
# No arguments. Configure with environment variables:
#   HFD_SCRIPT       Path to hfd.sh                    (default: ./hfd.sh)
#   HFD_EXTRA_ARGS   Extra args forwarded to hfd.sh    (e.g. "--hf_token xxx")
#   HF_ENDPOINT      Mirror endpoint                   (default: https://hf-mirror.com)
#   DRY_RUN          "true" to print commands only     (default: false)
#
# Example:
#   ./find_gitkeep_and_download.sh
#   DRY_RUN=true ./find_gitkeep_and_download.sh

set -euo pipefail

HFD_SCRIPT="${HFD_SCRIPT:-./hfd.sh}"
HFD_EXTRA_ARGS="${HFD_EXTRA_ARGS:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DRY_RUN="${DRY_RUN:-false}"
export HF_ENDPOINT

die() { echo "Error: $*" >&2; exit 1; }

[[ -x "$HFD_SCRIPT" ]] || die "hfd.sh not found or not executable at: $HFD_SCRIPT
  Get it with:
    curl -L https://hf-mirror.com/hfd.sh -o hfd.sh && chmod +x hfd.sh"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "Scanning for .gitkeep files under */models/*/*/ and */datasets/*/*/ ..."
echo

count=0
while IFS= read -r -d '' gitkeep; do
    # Strip leading ./
    gk="${gitkeep#./}"

    # Pattern breakdown:
    #   (.*/)?          optional leading path (greedy → picks the deepest valid match)
    #   (models|datasets)/   the HF category folder
    #   ([^/]+)/([^/]+)      username / reponame
    #   (/(.*))?             optional inner path inside the repo
    #   /\.gitkeep$          the marker file
    if [[ "$gk" =~ ^(.*/)?(models|datasets)/([^/]+)/([^/]+)(/(.*))?/\.gitkeep$ ]]; then
        prefix="${BASH_REMATCH[1]:-}"
        kind="${BASH_REMATCH[2]}"
        user="${BASH_REMATCH[3]}"
        repo="${BASH_REMATCH[4]}"
        inner="${BASH_REMATCH[6]:-}"
    else
        # .gitkeep not inside a models/ or datasets/ repo layout
        continue
    fi

    if [[ "$kind" == "models" ]]; then
        repo_type="model"
    else
        repo_type="dataset"
    fi

    repo_dir="./${prefix}${kind}/${user}/${repo}"

    # include pattern for this folder:
    #   - inner empty → we're at the repo root, fetch everything ("*")
    #   - inner non-empty → fetch everything under <inner>/ ("<inner>/*")
    if [[ -n "$inner" ]]; then
        pattern="${inner}/*"
    else
        pattern="*"
    fi

    printf '%s|%s|%s|%s\n' "$repo_type" "$user/$repo" "$repo_dir" "$pattern" >> "$TMP"
    count=$((count + 1))
done < <(find . -type d -name '.git' -prune -o -type f -name '.gitkeep' -print0)

if [[ "$count" -eq 0 ]]; then
    echo "No .gitkeep files found under */models/*/*/ or */datasets/*/*/."
    exit 0
fi

echo "Found $count .gitkeep file(s):"
echo

i=0
while IFS='|' read -r repo_type repo_id repo_dir pattern; do
    i=$((i + 1))

    dataset_flag=""
    [[ "$repo_type" == "dataset" ]] && dataset_flag="--dataset"

    echo "[$i/$count] $repo_type: $repo_id"
    echo "         repo dir : $repo_dir"
    echo "         include  : $pattern"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "         [DRY RUN] $HFD_SCRIPT $repo_id $dataset_flag --include '$pattern' --local-dir '$repo_dir' $HFD_EXTRA_ARGS"
        echo
        continue
    fi

    # shellcheck disable=SC2086
    "$HFD_SCRIPT" "$repo_id" \
        $dataset_flag \
        --include "$pattern" \
        --local-dir "$repo_dir" \
        $HFD_EXTRA_ARGS

    echo
done < "$TMP"

echo "Done."