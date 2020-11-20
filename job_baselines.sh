#!/bin/bash

if [ ! -d "/s0/ttmt001" ]
then
    space_req s0
    echo "Creating scratch space" 
fi

source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-gpu/bin/activate
#source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-cpu/bin/activate

python train_baseline.py --rnn_type gru --history_len 5 --gen_type greedy \
    --filename_note base-h5-gru --batch_size 64 --eval_batch_size 64

python train_baseline.py --rnn_type lstm --history_len 5 --gen_type greedy \
    --filename_note base-h5-lstm --batch_size 64 --eval_batch_size 64

python train_baseline.py --rnn_type gru --history_len 5 --gen_type beam --top_k 10 \
    --filename_note beam-h5-gru --batch_size 64 --eval_batch_size 64

#python train.py --rnn_type lstm --history_len 10 --gen_type beam --top_k 10 \
#    --filename_note baseline11 --batch_size 64 --eval_batch_size 64

#python train.py --rnn_type gru --history_len 10 --gen_type greedy \
#    --filename_note debug --n_epochs 2 \
#    --dial_encoder_hidden_dim 60 \
#    --decoder_hidden_dim 60 


