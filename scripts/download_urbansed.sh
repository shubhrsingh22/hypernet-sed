#!/usr/bin/env bash
# Download and extract the URBAN-SED v2.0.0 dataset.
# Usage: scripts/download_urbansed.sh /path/to/target_dir
set -euo pipefail

TARGET_DIR="${1:-data/URBAN-SED}"
URL="https://zenodo.org/records/1324404/files/URBAN-SED_v2.0.0.tar.gz?download=1"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

echo "Downloading URBAN-SED v2.0.0 (~6.5 GB) to ${TARGET_DIR} ..."
wget -c "${URL}" -O URBAN-SED_v2.0.0.tar.gz

echo "Extracting ..."
tar -xzf URBAN-SED_v2.0.0.tar.gz

echo "Done. Dataset root: ${TARGET_DIR}/URBAN-SED_v2.0.0"
