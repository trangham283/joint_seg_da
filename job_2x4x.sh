#!/bin/bash

if [ ! -d "/s0/ttmt001" ]
then
    space_req s0
    echo "Creating scratch space" 
fi

source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-gpu/bin/activate
#source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-cpu/bin/activate

for history_len in 1 3
do
    for gen_type in beam greedy
    do
        for freeze in "all" "pooler_only" "top_layer" 
        do
            python train.py \
                --freeze $freeze \
                --attr_embedding_dim 30 \
                --sent_encoder_hidden_dim 100 \
                --n_sent_encoder_layers 1 \
                --dial_encoder_hidden_dim 200 \
                --n_dial_encoder_layers 1 \
                --decoder_hidden_dim 200 \
                --n_decoder_layers 1 \
                --batch_size 64 --eval_batch_size 64 \
                --rnn_type gru \
                --history_len ${history_len} \
                --gen_type ${gen_type} \
                --top_k 10 \
                --filename_note 2x4x-bert-${gen_type}-hist${history_len}-ft${freeze}
        done
    done
done


