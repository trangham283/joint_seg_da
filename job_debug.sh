#!/bin/bash

if [ ! -d "/s0/ttmt001" ]
then
    space_req s0
    echo "Creating scratch space" 
fi

source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-gpu/bin/activate
#source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-cpu/bin/activate

############################################
# NOTE
# DON'T CHANGE THE PARAMS FOR COMPARISONS
# AND DEBUGS
############################################

gen_type=greedy
history_len=1
python train_sp.py \
    --model speech_attn_ed --debug True \
    --decode_max_len 45 \
    --d_pause_embedding 4 \
    --d_speech 12 \
    --downsample False \
    --num_conv 4 \
    --conv_sizes "5,10,25,50" \
    --feature_types "pitch,fb3,pause,pause_raw,word_dur" \
    --fixed_word_length 50 \
    --freeze "all" \
    --attr_embedding_dim 4 \
    --sent_encoder_hidden_dim 12 \
    --n_sent_encoder_layers 2 \
    --dial_encoder_hidden_dim 12 \
    --n_dial_encoder_layers 2 \
    --decoder_hidden_dim 12 \
    --n_decoder_layers 2 \
    --batch_size 32 --eval_batch_size 32 \
    --rnn_type gru \
    --history_len ${history_len} \
    --gen_type ${gen_type} \
    --top_k 10 --n_epochs 2 \
    --filename_note debug-sp \
    --save_model False --check_loss_after_n_step 10 --validate_after_n_step 20


