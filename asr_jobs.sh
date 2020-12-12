#!/bin/bash

source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-gpu/bin/activate
#source /g/ssli/transitory/ttmt001/envs/py3.6-torch1.7-cpu/bin/activate

history_len=1
posmode=absolute
poscomb=cat

model_name=tt5000
python predict_asr.py \
    --model_name ${model_name} \
    --model speech_bl \
    --seq_max_len 512 \
    --freeze "all" \
    --history_len ${history_len} \
    --num_conv 32 \
    --attr_embedding_dim 32 \
    --batch_size 32 \
    --pos_encoder_hidden_dim 32 \
    --pos_mode $posmode --pos_comb $poscomb 

model_name=tt5001
python predict_asr.py \
    --model_name ${model_name} \
    --model speech_bl \
    --seq_max_len 512 \
    --freeze "top_layer" \
    --history_len ${history_len} \
    --num_conv 32 \
    --attr_embedding_dim 32 \
    --batch_size 32 \
    --pos_encoder_hidden_dim 32 \
    --pos_mode $posmode --pos_comb $poscomb 

#model_name=sp5001
#python predict_asr.py \
#    --model_name ${model_name} \
#    --model speech_bl \
#    --seq_max_len 512 \
#    --freeze "top_layer" \
#    --history_len ${history_len} \
#    --d_pause_embedding 4 \
#    --d_speech 128 \
#    --downsample True \
#    --fixed_word_length 50 \
#    --num_conv 32 \
#    --conv_sizes "5,10,25,50" \
#    --feature_types "pitch,fb3,pause,pause_raw,word_dur" \
#    --attr_embedding_dim 32 \
#    --batch_size 32 \
#    --pos_encoder_hidden_dim 32 \
#    --pos_mode $posmode --pos_comb $poscomb 

#model_name=sp5000
#python predict_asr.py \
#    --model_name ${model_name} \
#    --model speech_bl \
#    --freeze "all" \
#    --seq_max_len 512 \
#    --history_len ${history_len} \
#    --d_pause_embedding 4 \
#    --d_speech 128 \
#    --downsample True \
#    --fixed_word_length 50 \
#    --num_conv 32 \
#    --conv_sizes "5,10,25,50" \
#    --feature_types "pitch,fb3,pause,pause_raw,word_dur" \
#    --attr_embedding_dim 32 \
#    --batch_size 32 \
#    --pos_encoder_hidden_dim 32 \
#    --pos_mode $posmode --pos_comb $poscomb 



## BERT add_pool_layer=False
#feats="pitch,fb3,pause_raw"
#history_len=3
#gen_type=beam
#model_name=sp30004
#python predict_asr.py \
#    --model_name ${model_name} \
#    --model speech_attn_ed \
#    --decode_max_len 100 \
#    --d_pause_embedding 12 \
#    --d_speech 128 \
#    --downsample True \
#    --fixed_word_length 50 \
#    --num_conv 32 \
#    --conv_sizes "5,10,25,50" \
#    --feature_types ${feats} \
#    --freeze "all" \
#    --attr_embedding_dim 64 \
#    --sent_encoder_hidden_dim 128 \
#    --n_sent_encoder_layers 2 \
#    --dial_encoder_hidden_dim 128 \
#    --n_dial_encoder_layers 2 \
#    --decoder_hidden_dim 128 \
#    --n_decoder_layers 2 \
#    --batch_size 32 \
#    --rnn_type gru \
#    --history_len ${history_len} \
#    --gen_type ${gen_type} \
#    --top_k 10 \
#    --filename_note ${model_name}
#
## BERT add_pool_layer=True
#model_name=sp10004
#feats="pitch,fb3,pause_raw"
#history_len=1
#gen_type=beam
#python predict_asr.py \
#    --model_name ${model_name} \
#    --model speech_attn_ed \
#    --decode_max_len 100 \
#    --d_pause_embedding 12 \
#    --d_speech 128 \
#    --downsample True \
#    --fixed_word_length 50 \
#    --num_conv 32 \
#    --conv_sizes "5,10,25,50" \
#    --feature_types ${feats} \
#    --freeze "all" \
#    --attr_embedding_dim 64 \
#    --sent_encoder_hidden_dim 128 \
#    --n_sent_encoder_layers 2 \
#    --dial_encoder_hidden_dim 128 \
#    --n_dial_encoder_layers 2 \
#    --decoder_hidden_dim 128 \
#    --n_decoder_layers 2 \
#    --batch_size 32 \
#    --rnn_type gru \
#    --history_len ${history_len} \
#    --gen_type ${gen_type} \
#    --top_k 10 \
#    --filename_note ${model_name}


# BERT add_pool_layer=True
#model_name=tt3000
#history_len=3
#gen_type=beam
#python predict_asr.py \
#    --model_name ${model_name} \
#    --decode_max_len 100 \
#    --freeze all \
#    --attr_embedding_dim 64 \
#    --sent_encoder_hidden_dim 128 \
#    --n_sent_encoder_layers 2 \
#    --dial_encoder_hidden_dim 128 \
#    --n_dial_encoder_layers 2 \
#    --decoder_hidden_dim 128 \
#    --n_decoder_layers 2 \
#    --batch_size 32 \
#    --rnn_type gru \
#    --history_len ${history_len} \
#    --gen_type ${gen_type} --top_k 10 \
#    --filename_note ${model_name}

# BERT add_pool_layer=False
#model_name=sp3000
#history_len=3
#gen_type=beam
#python predict_asr.py \
#    --model_name ${model_name} \
#    --model speech_attn_ed \
#    --d_pause_embedding 12 \
#    --d_speech 128 \
#    --downsample False \
#    --fixed_word_length 100 \
#    --num_conv 32 \
#    --conv_sizes "5,10,25,50" \
#    --feature_types "pitch,fb3,pause,pause_raw,word_dur" \
#    --decode_max_len 100 \
#    --freeze all \
#    --attr_embedding_dim 64 \
#    --sent_encoder_hidden_dim 128 \
#    --n_sent_encoder_layers 2 \
#    --dial_encoder_hidden_dim 128 \
#    --n_dial_encoder_layers 2 \
#    --decoder_hidden_dim 128 \
#    --n_decoder_layers 2 \
#    --batch_size 32 \
#    --rnn_type gru \
#    --history_len ${history_len} \
#    --gen_type ${gen_type} --top_k 10 \
#    --filename_note ${model_name}


# BERT add_pool_layer=True
#model_name=sp1000
#history_len=1
#gen_type=beam
#python predict_asr.py \
#    --model_name ${model_name} \
#    --model speech_attn_ed \
#    --d_pause_embedding 12 \
#    --d_speech 128 \
#    --downsample False \
#    --fixed_word_length 100 \
#    --num_conv 32 \
#    --conv_sizes "5,10,25,50" \
#    --feature_types "pitch,fb3,pause,pause_raw,word_dur" \
#    --decode_max_len 100 \
#    --freeze all \
#    --attr_embedding_dim 64 \
#    --sent_encoder_hidden_dim 128 \
#    --n_sent_encoder_layers 2 \
#    --dial_encoder_hidden_dim 128 \
#    --n_dial_encoder_layers 2 \
#    --decoder_hidden_dim 128 \
#    --n_decoder_layers 2 \
#    --batch_size 32 \
#    --rnn_type gru \
#    --history_len ${history_len} \
#    --gen_type ${gen_type} --top_k 10 \
#    --filename_note ${model_name}


# BERT add_pool_layer=True
#model_name=tt1000
#history_len=1
#gen_type=beam
#python predict_asr.py \
#    --model_name ${model_name} \
#    --decode_max_len 100 \
#    --freeze all \
#    --attr_embedding_dim 64 \
#    --sent_encoder_hidden_dim 128 \
#    --n_sent_encoder_layers 2 \
#    --dial_encoder_hidden_dim 128 \
#    --n_dial_encoder_layers 2 \
#    --decoder_hidden_dim 128 \
#    --n_decoder_layers 2 \
#    --batch_size 32 \
#    --rnn_type gru \
#    --history_len ${history_len} \
#    --gen_type ${gen_type} --top_k 10 \
#    --filename_note ${model_name}


