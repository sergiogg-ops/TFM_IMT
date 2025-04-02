# Interactive machine translation

This git contains code to train several LLMs and perform simulated sesions of interactive machine translation based in prefixes and segments. This is:

1. The system translates the sentence to the target language, offering an initial hypothesis
2. If the translation is good enough the "human" can stop the proccess. Also, one can say that the best solution is the result of concatenate all the validated segments.
3. The human can mark as many segments as needed. In the prefix based approach the only segment that one can mark is the one between the begining of the translation and the first incorrect word.
4. In the segment based approach the human can mark pairs of segments that should be one right after the other in the ideal translation.
5. The human can type some corrections by keyboard.
6. The system will try another hypothesis minding the restrictions that the human has introduced.

# Use
- `train.py` can be used to fine-tune several LLMs to the IMT task, using the pytorch lightning framework.
- `utils/unwrapp.py` can be used to extract the fine-tuned models from the lightning envelope, and obtain the weights of the models.
- `bleu_ter.py` can be used to evaluate the models in a classic machine translation task.
- `imt.py` can be used to evaluate the models in a simulated IMT session with some dataset. Both segment-based and prefix-based protocols are available.
- `model.py` is a library script that contains utilities for LLMs management.
- `restriction.py` is a library script that contains utilities for constraining the output of the models in an IMT session.

There are some analysis and visualization utilities in the `utils` folder too.

# Supported LLMs
Here are the models that can be used directly with this repository, their scientific articles and the base models in huggingface:
| Model | Huggingface |
|-------|-------------|
| [mBART](https://arxiv.org/abs/2001.08210) | facebook/mbart-large-50-many-to-many-mmt |
| [M2M](https://arxiv.org/abs/2010.11125)   | facebook/m2m100_418M |
| [Flan-T5](https://arxiv.org/abs/2210.11416) | google/flan-t5-base |
| [NLLB](https://arxiv.org/abs/2207.04672) | facebook/nllb-200-distilled-600M |
| [Llama 3.2](https://arxiv.org/abs/2407.21783) | meta-llama/Llama-3.2-1B-Instruct |
| [EuroLLM](https://arxiv.org/abs/2409.16235) | utter-project/EuroLLM-1.7B-Instruct |
| [Gemma 3](https://arxiv.org/abs/2503.19786) | google/gemma-3-1b-it |

# Citation
If you use this repository, please cite:
```
@inproceedings{gomez2024interactive,
  title={Interactive Machine Translation with Large Language Models in Low Resources Languages},
  author={G{\'o}mez, Sergio and Domingo, Miguel and Casacuberta, Francisco},
  booktitle={Proc. IberSPEECH 2024},
  pages={66--70},
  year={2024}
}
```
