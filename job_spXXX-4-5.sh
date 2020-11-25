#!/bin/bash

if [ ! -d "/s0/ttmt001" ]
then
    space_req s0
    echo "Creating scratch space" 
fi

source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-gpu/bin/activate
#source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-cpu/bin/activate

gen_type=beam

for history_len in 1 3
do
    python train_sp.py \
    --model speech_attn_ed \
    --decode_max_len 100 \
    --d_pause_embedding 4 \
    --d_speech 256 \
    --downsample False \
    --num_conv 32 \
    --conv_sizes "5,10,25,50" \
    --feature_types "pitch,fb3,pause,pause_raw,word_dur" \
    --fixed_word_length 100 \
    --freeze "all" \
    --attr_embedding_dim 64 \
    --sent_encoder_hidden_dim 128 \
    --n_sent_encoder_layers 2 \
    --dial_encoder_hidden_dim 256 \
    --n_dial_encoder_layers 2 \
    --decoder_hidden_dim 256 \
    --n_decoder_layers 2 \
    --batch_size 32 --eval_batch_size 32 \
    --rnn_type gru \
    --history_len ${history_len} \
    --gen_type ${gen_type} \
    --top_k 10 --n_epochs 15 \
    --filename_note sp${history_len}004
done

for history_len in 1 3
do
    python train_sp.py \
    --model speech_attn_ed \
    --decode_max_len 100 \
    --d_pause_embedding 4 \
    --d_speech 256 \
    --downsample False \
    --num_conv 64 \
    --conv_sizes "5,10,25,50" \
    --feature_types "pitch,fb3,pause,pause_raw,word_dur" \
    --fixed_word_length 100 \
    --freeze "all" \
    --attr_embedding_dim 64 \
    --sent_encoder_hidden_dim 128 \
    --n_sent_encoder_layers 2 \
    --dial_encoder_hidden_dim 256 \
    --n_dial_encoder_layers 2 \
    --decoder_hidden_dim 256 \
    --n_decoder_layers 2 \
    --batch_size 32 --eval_batch_size 32 \
    --rnn_type gru \
    --history_len ${history_len} \
    --gen_type ${gen_type} \
    --top_k 10 --n_epochs 15 \
    --filename_note sp${history_len}005
done


