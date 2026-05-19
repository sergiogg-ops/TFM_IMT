#!/bin/bash

# python train.py -src gl -trg en -dir hplt/gl-en/ -model gemma -lora -bs 2
# python train.py -src en -trg gl -dir hplt/gl-en/ -model gemma -lora -bs 2
# python train.py -src gl -trg en -dir hplt/gl-en/ -model eurollm -lora -bs 4
# python train.py -src en -trg gl -dir hplt/gl-en/ -model eurollm -lora -bs 4

# python train.py -src sw -trg en -dir hplt/sw-en/ -model gemma -lora -bs 2
# python train.py -src en -trg sw -dir hplt/sw-en/ -model gemma -lora -bs 2
# python train.py -src sw -trg en -dir hplt/sw-en/ -model eurollm -lora -bs 4
# python train.py -src en -trg sw -dir hplt/sw-en/ -model eurollm -lora -bs 4

# python train.py -src ca -trg en -dir hplt/ca-en/ -model gemma -lora -bs 2
# python train.py -src en -trg ca -dir hplt/ca-en/ -model gemma -lora -bs 2
# python train.py -src ca -trg en -dir hplt/ca-en/ -model eurollm -lora -bs 4
# python train.py -src en -trg ca -dir hplt/ca-en/ -model eurollm -lora -bs 4

# -------------------------------------------------------------------------------

#python bleu_ter.py -src en -trg gl -dir hplt/gl-en/ -p test -model gemma_engl/ -model_name gemma -b 1
#python bleu_ter.py -src en -trg gl -dir hplt/gl-en/ -p test -model eurollm_engl/ -model_name eurollm -b 1
#python bleu_ter.py -src gl -trg en -dir hplt/gl-en/ -p test -model gemma_glen/ -model_name gemma -b 1
#python bleu_ter.py -src gl -trg en -dir hplt/gl-en/ -p test -model eurollm_glen/ -model_name eurollm -b 1

#python bleu_ter -src sw -trg en -dir hplt/sw-en/ -p test -model gemma_swen/ -model_name gemma -b 1
#python bleu_ter -src sw -trg en -dir hplt/sw-en/ -p test -model eurollm_swen/ -model_name eurollm -b 1
#python bleu_ter -src en -trg sw -dir hplt/sw-en/ -p test -model gemma_ensw/ -model_name gemma -b 1
#python bleu_ter -src en -trg sw -dir hplt/sw-en/ -p test -model eurollm_ensw/ -model_name eurollm -b 1

#python bleu_ter.py -src en -trg ca -dir hplt/ca-en/ -p test -model gemma_caen/ -model_name gemma -b 1
#python bleu_ter.py -src en -trg ca -dir hplt/ca-en/ -p test -model eurollm_caen/ -model_name eurollm -b 1
#python bleu_ter.py -src ca -trg en -dir hplt/ca-en/ -p test -model gemma_caen/ -model_name gemma -b 1
#python bleu_ter.py -src ca -trg en -dir hplt/ca-en/ -p test -model eurollm_caen/ -model_name eurollm -b 1

# -------------------------------------------------------------------------------

#python imt.py -src gl -trg en -dir hplt/gl-en/ -model gemma_glen/ -out pb_gemma  -model_name gemma -p test
#python imt.py -src gl -trg en -dir hplt/gl-en/ -model gemma_glen/ -out sb_gemma  -model_name gemma -p test -seg
#python imt.py -src gl -trg en -dir hplt/gl-en/ -model eurollm_glen/ -out pb_eurollm  -model_name eurollm -p test
#python imt.py -src gl -trg en -dir hplt/gl-en/ -model eurollm_glen/ -out sb_eurollm  -model_name eurollm -p test -seg

#python imt.py -src en -trg gl -dir hplt/gl-en/ -model gemma_engl/ -out pb_gemma  -model_name gemma -p test
#python imt.py -src en -trg gl -dir hplt/gl-en/ -model gemma_engl/ -out sb_gemma  -model_name gemma -p test -seg
#python imt.py -src en -trg gl -dir hplt/gl-en/ -model eurollm_engl/ -out pb_eurollm  -model_name eurollm -p test
#python imt.py -src en -trg gl -dir hplt/gl-en/ -model eurollm_engl/ -out sb_eurollm  -model_name eurollm -p test -seg

# python imt.py -src sw -trg en -dir hplt/sw-en/ -model gemma_swen/ -out pb_gemma  -model_name gemma -p test
# python imt.py -src sw -trg en -dir hplt/sw-en/ -model gemma_swen/ -out sb_gemma  -model_name gemma -p test -seg
# python imt.py -src sw -trg en -dir hplt/sw-en/ -model eurollm_swen/ -out pb_eurollm  -model_name eurollm -p test
# python imt.py -src sw -trg en -dir hplt/sw-en/ -model eurollm_swen/ -out sb_eurollm  -model_name eurollm -p test -seg

# python imt.py -src en -trg sw -dir hplt/sw-en/ -model gemma_ensw/ -out pb_gemma  -model_name gemma -p test
# python imt.py -src en -trg sw -dir hplt/sw-en/ -model gemma_ensw/ -out sb_gemma  -model_name gemma -p test -seg
# python imt.py -src en -trg sw -dir hplt/sw-en/ -model eurollm_ensw/ -out pb_eurollm  -model_name eurollm -p test
# python imt.py -src en -trg sw -dir hplt/sw-en/ -model eurollm_ensw/ -out sb_eurollm  -model_name eurollm -p test -seg

python imt.py -src ca -trg en -dir hplt/ca-en/ -model gemma_caen/ -out pb_gemma  -model_name gemma -p test
python imt.py -src ca -trg en -dir hplt/ca-en/ -model gemma_caen/ -out sb_gemma  -model_name gemma -p test -seg
# python imt.py -src ca -trg en -dir hplt/ca-en/ -model eurollm_caen/ -out pb_eurollm  -model_name eurollm -p test
# python imt.py -src ca -trg en -dir hplt/ca-en/ -model eurollm_caen/ -out sb_eurollm  -model_name eurollm -p test -seg

# python imt.py -src en -trg ca -dir hplt/ca-en/ -model gemma_enca/ -out pb_gemma  -model_name gemma -p test
# python imt.py -src en -trg ca -dir hplt/ca-en/ -model gemma_enca/ -out sb_gemma  -model_name gemma -p test -seg
# python imt.py -src en -trg ca -dir hplt/ca-en/ -model eurollm_enca/ -out pb_eurollm  -model_name eurollm -p test
# python imt.py -src en -trg ca -dir hplt/ca-en/ -model eurollm_enca/ -out sb_eurollm  -model_name eurollm -p test -seg
