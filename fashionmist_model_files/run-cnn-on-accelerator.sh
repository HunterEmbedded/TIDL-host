#!/bin/bash
# runtime execution of model on beagley-ai

export TIDL_RT_DEBUG=1
export TIDL_RT_PERFSTATS=1
export VX_ZONE_MASK=0xffffffff
export ONNXRT_LOG_SEVERITY_LEVEL=0

python3 onnxrt_ep_fashionmist.py -m cl-ort-fashioncnn28