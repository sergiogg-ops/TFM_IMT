# Interactive machine translation

This git contains code to train several LLMs and perform simulated sesions of interactiva machine translation based in prefixes and segments. This is:

1. The system translates the sentence to the target language, offering an initial hypothesis
2. If the translation is good enough the "human" can stop the proccess. Also, one can say that the best solution is the result of concatenate all the validated segments.
3. The human can mark as many segments as needed. In the prefix based approach the only segment that one can mark is the one between the begining of the translation and the first incorrect word.
4. In the segment based approach the human can mark pairs of segments that should be one right after the other in the ideal translation.
5. The human can type some corrections by keyboard.
6. The system will try another hypothesis minding the restrictions that the human has introduced.

To train a model in a specific translation direction one can use the `train.py` script. That model can be evaluated with the scripts in the `eval` folder after being extracted with the `utils/unwrapp.py` script. The `eval/bleu_ter.py` script evaluates the model over a classical translation task. The `eval/imt.py` permits evaluating the models over an interactive machine translation task with a simulated user.

There are some analysis and visualization utilities in the utils folder. 