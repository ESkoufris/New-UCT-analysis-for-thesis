#!/usr/bin/env bash
set -euo pipefail

# Base directory
OUTDIR="new_figures/UCT/500k"

# Transition/reward cases
CASES=(
  "trans_det-rew_det"
  "trans_det-rew_sto"
  "trans_sto-rew_det"
  "trans_sto-rew_sto"
)

# A and H values
A_LIST=(2 3)
H_LIST=(3 4)

# Create all folders
for CASE in "${CASES[@]}"; do
  for A in "${A_LIST[@]}"; do
    for H in "${H_LIST[@]}"; do
      DIR="${OUTDIR}/${CASE}/A${A}/H${H}"
      mkdir -p "$DIR"
      echo "Created: $DIR"
    done
  done
done

echo "✅ Folder structure created."