#!/bin/bash
# script to setup pytorch environment on host to train model

# now setup virtual env for pytorch

VENV=".venv-pytorch"

mkdir -p pytorch-cnn
cd pytorch-cnn || exit


if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import torch" 2>/dev/null; then
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install torch torchvision onnx onnxruntime onnxscript
fi

source "$VENV/bin/activate"

# Now run the fashionmist database training and then export trained model to onnx
python3 train_export_fashionmist_cnn.py