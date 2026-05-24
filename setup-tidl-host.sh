#!/bin/bash
# script to setup TIDL HOST environment targetting BeagleY-AI


# clone edgeai tools
if [ ! -d edgeai-tidl-tools ]; then
        git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git || exit
        cd edgeai-tidl-tools
        git checkout 11_00_06_00
fi 


# now setup for BeagleY-AI
python3 -m venv .venv-tidl
source .venv-tidl/bin/activate

# setup the system
export SOC=am67a
source ./setup.sh
# set up the env variables for specific board
source ./setup_env.sh ${SOC}





