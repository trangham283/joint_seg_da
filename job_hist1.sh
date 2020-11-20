#!/bin/bash

if [ ! -d "/s0/ttmt001" ]
then
    space_req s0
    echo "Creating scratch space" 
fi

source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-gpu/bin/activate
#source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-cpu/bin/activate

python train.py --rnn_type gru --history_len 1 --gen_type beam --top_k 10 \
    --filename_note bert_hist1 --batch_size 64 --eval_batch_size 64

