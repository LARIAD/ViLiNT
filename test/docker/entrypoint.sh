#!/bin/bash
set -e

REPO_DIR="/workspace/multimodal-navigation-transformer"

cd "${REPO_DIR}"

conda run --no-capture-output -n lint pip install -e diffusion
conda run --no-capture-output -n lint pip install -e train

exec /bin/bash -i
