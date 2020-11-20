#!/bin/bash

if [ ! -d "/s0/ttmt001" ]
then
    space_req s0
    echo "Creating scratch space" 
fi

source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-gpu/bin/activate
#source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-cpu/bin/activate

history_len=10
n_epochs=10
gen_type=beam
python train_baseline.py \
    --use_pretrained_word_embedding False \
    --word_embedding_dim 200 \
    --attr_embedding_dim 30 \
    --sent_encoder_hidden_dim 100 \
    --n_sent_encoder_layers 1 \
    --dial_encoder_hidden_dim 200 \
    --n_dial_encoder_layers 1 \
    --decoder_hidden_dim 200 \
    --n_decoder_layers 1 \
    --batch_size 30 --eval_batch_size 30 \
    --rnn_type gru \
    --history_len ${history_len} \
    --gen_type ${gen_type} \
    --top_k 20 \
    --filename_note csl2019
