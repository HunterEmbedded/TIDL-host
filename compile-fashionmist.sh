#!/bin/bash

BaseDir="$PWD"
echo "BaseDir $BaseDir"
# script to compile the fashionmist model for BeagleY-AI
# must be run in venv, so do run this first
# source edgeai-tidl-tools/.venv-tidl/bin/activate


if [ ! -n "$VIRTUAL_ENV" ]; then
    echo "Must be run in a virtual environment. Please run:"
    echo "source edgeai-tidl-tools/.venv-tidl/bin/activate"
fi

# Copy in the exported model from pytorch
cd edgeai-tidl-tools || exit

# set up paths 
source ./setup_env.sh am67a

mkdir -p exported-pytorch-models/public || exit
mkdir -p exported-pytorch-model-artifacts/cl-ort-fashioncnn28 || exit

cp ../pytorch-cnn/model_cnn_shaped.onnx exported-pytorch-models/public/ || exit

# copy in the customised files to handle that model
cp $BaseDir/fashionmist_model_files/model_configs_fashionmist.py examples/osrt_python/ || exit
cp $BaseDir/fashionmist_model_files/onnxrt_ep_fashionmist.py examples/osrt_python/ort/ || exit

# and compile
cd examples/osrt_python/ort || exit
python3 onnxrt_ep_fashionmist.py -c -m cl-ort-fashioncnn28


# and execute on CPU as a test
python3 onnxrt_ep_fashionmist.py -d -m cl-ort-fashioncnn28
