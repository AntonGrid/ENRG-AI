#!/usr/bin/env bash
# Connect the workspace projects to ENRG-AI via symlinks.
#
# Projects are deliberately NOT committed (see .gitignore `projects/*`):
# every developer links the local clones they work with. The paths below
# assume the standard AXIS workspace layout:
#
#     ~/Axis-workspace/ENRG-AI
#     ~/Axis-workspace/ENRG
#     ~/Axis-workspace/enrg-landing
#     ~/Axis-workspace/Axis-core
#
# Usage:  bash scripts/link_projects.sh   (from the ENRG-AI repo root)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${ROOT}/../.." && pwd)"

link() {
    local name="$1" target="$2"
    if [ -e "${ROOT}/projects/${name}" ] && [ ! -L "${ROOT}/projects/${name}" ]; then
        echo "skip: projects/${name} exists and is not a symlink"
        return
    fi
    rm -rf "${ROOT}/projects/${name}"
    ln -s "${target}" "${ROOT}/projects/${name}"
    echo "linked projects/${name} -> ${target}"
}

link "ENRG"        "${WORKSPACE}/ENRG"
link "enrg-landing" "${WORKSPACE}/enrg-landing"
link "Axis-core"    "${WORKSPACE}/Axis-core"

echo "Done. Configure paths in agent/config.py if your layout differs."
