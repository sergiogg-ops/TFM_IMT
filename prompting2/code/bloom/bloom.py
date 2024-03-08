from transformers import BloomTokenizerFast, BloomForCausalLM
from nltk.tokenize import wordpunct_tokenize
import torch
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def read_file(name):
	file_r = open(name, 'r')
	lines = file_r.read().splitlines()
	file_r.close()
	return lines

def check_prefix(target, hyp):
	correction = False
	target = wordpunct_tokenize(target)
	hyp = wordpunct_tokenize(hyp)
	prefix = []

	for i in range(len(target)):
		if len(hyp)<=i:
			correction = True
			prefix.append(target[i])
			break
		elif target[i] == hyp[i]:
			prefix.append(target[i])
		else:
			correction = True
			prefix.append(target[i])
			break
	return ' '.join(prefix), correction

#|====================== 
#|    SOURCE FILE
#| >src_lines
#|====================== 
#src_lines = read_file('/dataset/test-Paco/europarl-v7.es-en-test-hidden.en')
src_lines = read_file('../europarl-inmt/es-en/test.en')

#|====================== 
#|    OUTPUT FILE
#| >file_w
#|======================
file_w = open('output.es', 'w')

#|====================== 
#|    LOAD MODELS
#| >tokenizer
#| >model
#|======================
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7", device_map="auto")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b7", device_map="auto")

#|======================
#|   READ THE PROMPT
#| >header_prompt
#| >header_size
#|======================
file_r = open('prompting2/code/bloom/prompt1.txt', 'r')
header_prompt = file_r.read()
header_size = len(tokenizer(header_prompt, return_tensors="pt").input_ids[0])-1
file_r .close()

for i in range(0, len(src_lines)):


	source = ' '.join(wordpunct_tokenize(src_lines[i]))
	print("Sentence {0}:\n\t{1}".format(i+1, source))

	prompt = header_prompt + "{0} ::: ".format(src_lines[i])

	input_ids = tokenizer(prompt, return_tensors="pt").to(device)
	generated_tokens = model.generate(**input_ids, max_new_tokens=100)
	output = tokenizer.decode(generated_tokens[0][header_size:], skip_special_tokens=True)

	if output != '':
		output = output.splitlines()[0].split(':::')[1].lstrip(' ')
	print("\t{0}\n".format( output))

	file_w.write('{}\n'.format(output))
