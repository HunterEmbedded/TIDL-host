#!/bin/bash
# script to setup TIDL HOST environment targetting BeagleY-AI


# clone edgeai tools
if [ ! -d edgeai-tidl-tools ]; then
        git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git || exit
        cd edgeai-tidl-tools
        git checkout rel_11_00
fi 

# now setup for BeagleY-AI
python3 -m venv .venv-tidl
source .venv-tidl/bin/activate
python -m pip install --upgrade pip setuptools wheel



