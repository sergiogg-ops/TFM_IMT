#!/bin/bash

conda activate tfm

python segment_based.py -model mbart_enes -src en -trg es -dir europarl-inmt/es-en -model_name mbart
echo "MBART:"
tail -n 1 europarl-inmt/es-en/imt_mbart.es

python segment_based.py -model m2m_enes -src en -trg es -dir europarl-inmt/es-en -model_name m2m
echo "M2M:"
tail -n 1 europarl-inmt/es-en/imt_m2m.es

python segment_based.py -model flant5_enes -src en -trg es -dir europarl-inmt/es-en -model_name flant5
echo "Flant5:"
tail -n 1 europarl-inmt/es-en/imt_flant5.es