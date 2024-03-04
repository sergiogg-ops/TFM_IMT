"""
Prefix-Based Approach with T5
Testing with English-French
"""

from transformers import T5Tokenizer, MT5ForConditionalGeneration, T5ForConditionalGeneration
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.tokenize import wordpunct_tokenize
import re

wordTokenizer = TreebankWordTokenizer()

def read_file(name):
	file_r = open(name, 'r')
	lines = file_r.read().splitlines()
	file_r.close()
	return lines

def get_prefix(s1, s2):
	prefix = []
	correction = 0
	for i in range(len(s1)):
		if i < len(s2):
			if s1[i] == s2[i]:
				prefix.append(s1[i])
			else:
				break
		else:
			correction = 1
			break

	if len(prefix)<len(s2):
		correction = 2
		prefix.append(ref[len(prefix)])

	return prefix, correction

src_lines = read_file('/dataset/europarl-tst/fr-en/tst.en')
trg_lines = read_file('/dataset/europarl-tst/fr-en/tst.fr')

file_out = open('/dataset/europarl-tst/fr-en/pre_t5.fr', 'w')

tokenizer = T5Tokenizer.from_pretrained("t5-large", device_map="auto")
model = T5ForConditionalGeneration.from_pretrained("t5-large", device_map="auto")

def translate(c_sentence):
	prompt = "English: {0}\n French:\n".format(c_sentence)

	input_ids = tokenizer(prompt, return_tensors="pt").input_ids
	generated_tokens = model.generate(input_ids)
	c_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

	return ' '.join(wordTokenizer.tokenize(c_translation)).replace('``', '"')

def prefix_based(c_sentence, c_prefix, ref):
	prompt = "English: {0}\n French:\n".format(c_sentence)

	# Anyadir el prefijo al prompt
	prompt += ' '.join(c_prefix)
	prompt += ' <extra_id_0>'

	# Generar el output con el modelo
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids
	generated_tokens = model.generate(input_ids)
	c_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)

	# Procesar el tag especial
	c_output = c_output.replace('</s>', '')
	elements = re.split('<extra_id_[0-99]*>', c_output)[1:]

	# Montar la traduccion
	c_translation = ' '.join(c_prefix)
	if elements:
		c_translation += ' ' + elements[0]

	return c_translation

total_words = 0
total_ws = 0
total_ma = 0
for i in range(0, len(src_lines)):
	c_sentence  = src_lines[i]
	c_reference = trg_lines[i]
	n_ite = 0
	print("\nProcessing sentence {0}: {1}\nReference: {2}".format(i, c_sentence, c_reference))

	c_translation = translate(c_sentence)
	print("\tITE {0}: {1}".format(n_ite, c_translation))
	n_ite += 1

	hyp = c_translation.split()
	ref = c_reference.split()

	mouse_actions = 0
	word_strokes = 0
	n_words = len(ref)
	while (hyp[:len(ref)] != ref):
		prefix, correction = get_prefix(hyp, ref)
		c_translation = prefix_based(c_sentence, prefix, ref)

		if correction == 0:
			mouse_actions += 1
		elif correction == 1:
			mouse_actions += 1
		elif correction == 2:
			mouse_actions += 1
			word_strokes += 1

		print("\tITE {0}: {1}".format(n_ite, c_translation))
		n_ite += 1
		hyp = c_translation.split()

	print("WSR: {0:.4f} MAR: {1:.4f}".format(word_strokes/n_words, mouse_actions/n_words))
	total_words += n_words
	total_ws += word_strokes
	total_ma += mouse_actions

	if (i+1)%10 == 0:
		print("T_WSR: {0:.4f} T_MAR: {1:.4f}".format(total_ws/total_words, total_ma/total_words))
		file_out.write("{2} T_WSR: {0:.4f} T_MAR: {1:.4f}\n".format(total_ws/total_words, total_ma/total_words, i))
file_out.write("END T_WSR: {0:.4f} T_MAR: {1:.4f}\n".format(total_ws/total_words, total_ma/total_words))
file_out.close()
