#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y curl ca-certificates libgl1 libglib2.0-0 unzip
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-coral-export.txt
python -c "import ultralytics,tensorflow,onnx2tf,ai_edge_litert,flatbuffers,numpy;print(ultralytics.__version__,tensorflow.__version__,numpy.__version__)"
python -m pip freeze > freeze_nouvel_env.txt
