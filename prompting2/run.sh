#!/bin/sh

# BASIC
#docker run -it --rm --gpus device=0 -v $PWD/code:/code -v $PWD/dataset:/dataset hf bash -c "cd /code/mbart && python3 trn_mbart.py"
#docker run -it --rm --gpus device=0 -v $PWD/code:/code -v $PWD/dataset:/dataset hf bash #-c "cd /code/mbart && python3 mbart.py -src en -trg es -dir es-en -model models/europarl_enes/checkpoint-613500/"

#docker run --rm --gpus device=0 -v $PWD/code:/code -v $PWD/dataset:/dataset -v /data/annamar/mbart:/models hf bash -c "cd /code/t5 && python3 sb_mt5.py -src de -trg en -dir /dataset/europarl-inmt/de-en/ -model /models/europarl_deen/checkpoint-593500/ 0> log0_deen 1> log1_deen 2> log2_deen"
#docker run --rm --gpus device=1 -v $PWD/code:/code -v $PWD/dataset:/dataset -v /data/annamar/mbart:/models hf bash -c "cd /code/t5 && python3 sb_mt5.py -src en -trg de -dir /dataset/europarl-inmt/de-en/ -model /models/europarl_ende/checkpoint-356000/ 0> log0_ende 1> log1_ende 2> log2_ende" 
docker run --rm --gpus device=0 -v $PWD/code:/code -v $PWD/dataset:/dataset -v /data/annamar/mt5:/models hf bash -c "cd /code/t5 && python3 sb_mt5.py -src es -trg en -dir /dataset/europarl-inmt/es-en/ -model /models/europarl_esen/checkpoint-46000/ 0> log0_esen 1> log1_esen 2> log2_esen"
#docker run --rm --gpus device=1 -v $PWD/code:/code -v $PWD/dataset:/dataset -v /data/annamar/mbart:/models hf bash -c "cd /code/t5 && python3 sb_mt5.py -src en -trg es -dir /dataset/europarl-inmt/es-en/ -model /models/europarl_enes/checkpoint-613000/ 0> log0_enes 1> log1_enes 2> log2_enes"
#docker run --rm --gpus device=0 -v $PWD/code:/code -v $PWD/dataset:/dataset -v /data/annamar/mbart:/models hf bash -c "cd /code/t5 && python3 sb_mt5.py -src fr -trg en -dir /dataset/europarl-inmt/fr-en/ -model /models/europarl_fren/checkpoint-372000/ 0> log0_fren 1> log1_fren 2> log2_fren"
#docker run --rm --gpus device=1 -v $PWD/code:/code -v $PWD/dataset:/dataset -v /data/annamar/mbart:/models hf bash -c "cd /code/t5 && python3 sb_mt5.py -src en -trg fr -dir /dataset/europarl-inmt/fr-en/ -model /models/europarl_enfr/checkpoint-372000/ 0> log0_enfr 1> log1_enfr 2> log2_enfr"
