#!/bin/bash
# copy host built files to BeagleY-AI

TARGET="10.159.1.44"


# Copy the test files to target if they are not already there
if ! ssh root@${TARGET} "[ -d /root/beagley-ai-tidl/fashionmist_test_images ]"; then
    scp -r fashionmist_test_images root@${TARGET}:/root/beagley-ai-tidl
fi

cd edgeai-tidl-tools || exit
# copy artifacts
scp -r exported-pytorch-models root@${TARGET}:/root/beagley-ai-tidl/edgeai-tidl-tools/
scp -r exported-pytorch-model-artifacts root@${TARGET}:/root/beagley-ai-tidl/edgeai-tidl-tools/

# and the customised model_configs and script
scp  examples/osrt_python/model_configs_fashionmist.py root@${TARGET}:/root/beagley-ai-tidl/edgeai-tidl-tools/examples/osrt_python
scp  examples/osrt_python/ort/onnxrt_ep_fashionmist.py root@${TARGET}:/root/beagley-ai-tidl/edgeai-tidl-tools/examples/osrt_python/ort

# copy runtime bash script
scp ../fashionmist_model_files/run-cnn-on-accelerator.sh root@${TARGET}:/root/beagley-ai-tidl/edgeai-tidl-tools/examples/osrt_python/ort

