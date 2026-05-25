#!/bin/bash
# runtime execution of model on beagley-ai

# some run time debug options
#export TIDL_RT_DEBUG=1
#export TIDL_RT_PERFSTATS=1
#export VX_ZONE_MASK=0xffffffff
#export ONNXRT_LOG_SEVERITY_LEVEL=0


# run CPU execution
python3 onnxrt_ep_fashionmist.py -d -m cl-ort-fashioncnn28
# then DSP
python3 onnxrt_ep_fashionmist.py -m cl-ort-fashioncnn28