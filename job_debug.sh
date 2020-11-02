#!/bin/bash

if [ ! -d "/s0/ttmt001" ]
then
    space_req s0
    echo "Creating scratch space" 
fi

#source /g/ssli/transitory/ttmt001/envs/py3.6-transformers-gpu/bin/activate
source /g/ssli/transitory/ttmt001/envs/py3.6-transformers-cpu/bin/activate

