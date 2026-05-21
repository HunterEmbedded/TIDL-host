# Copyright (c) 2018-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import platform
from pathlib import Path
from config_utils import create_model_config

BASE_DIR = Path(__file__).parent
models_base_path = BASE_DIR / '../../exported-pytorch-models/public/'
artifacts_base_path = BASE_DIR / '../../exported-pytorch-model-artifacts'

if platform.machine() == 'aarch64':
    numImages = 100
else : 
    import requests
    import onnx
    numImages = 3

models_configs = {
    ############ onnx models ##########
    "cl-ort-fashioncnn28": create_model_config(
    task_type="classification",
    source=dict(
        model_url="",
        infer_shape=False,
    ),
    preprocess=dict(
        resize=28,
        crop=28,
        data_layout="NCHW",
        resize_with_pad=False,
        reverse_channels=False,
    ),
    session=dict(
        session_name="onnxrt",
        model_path=os.path.join(models_base_path, "model_cnn_shaped.onnx"),
        artifacts_folder=os.path.join(artifacts_base_path, "cl-ort-fashioncnn28"),
        input_mean=[0.0],
        input_scale=[1.0],
        input_optimization=False,
    ),
    postprocess=dict(),
    extra_info=dict(
        num_images=1,
        num_classes=10,
        input_images=BASE_DIR / '../../../fashionmist_test_images'
    ),
)
}

