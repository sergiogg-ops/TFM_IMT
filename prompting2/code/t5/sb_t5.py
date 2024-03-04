"""
Segment-Based Approach with T5
Testing with English-French
"""

from transformers import T5Tokenizer, MT5ForConditionalGeneration, T5ForConditionalGeneration
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import wordpunct_tokenize
import torch
import re

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def read_file(name):
	file_r = open(name, 'r')
	lines = file_r.read().splitlines()
	file_r.close()
	return lines

def longest_common_substring(s1, s2):
	m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
	longest, x_longest, y_longest = 0, 0, 0
	for x in range(1, 1+len(s1)):
		for y in range(1, 1+len(s2)):
			if s1[x-1] == s2[y-1]:
				m[x][y] =  m[x-1][y-1] +1
				if m[x][y] >= longest:
					longest = m[x][y]
					x_longest = x
					y_longest = y
			else:
				m[x][y] = 0
	return (s1[x_longest-longest:x_longest], x_longest-longest, y_longest-longest)

def get_segments(s1, s2, s1_offset=0, s2_offset=0):
	if s1==[] or s2==[]:
		return [], []

	com, s1_start, s2_start = longest_common_substring(s1, s2)
	len_common = len(com)
	if len_common == 0:
		return [], []

	s1_before = s1[:s1_start]
	s2_before = s2[:s2_start]
	s1_after = s1[s1_start+len_common:]
	s2_after = s2[s2_start+len_common:]
	before = get_segments(s1_before, s2_before, s1_offset, s2_offset)
	after  = get_segments(s1_after, s2_after, s1_offset+s1_start+len_common, s2_offset+s2_start+len_common)

	return (before[0] + [[s1_offset+s1_start, com]] + after[0],
			before[1] + [[s2_offset+s2_start, com]] + after[1])

def merge_segments(c_segments):
	segments = []
	correction = 0

	for i in c_segments:
		if segments == []:
			segments.append(i)
		else:
			l_segment = segments[-1]

			if l_segment[0]+len(l_segment[1])==i[0]:
				correction += 1
				l_segment[1] += i[1]
			else:
				segments.append(i)

	return segments, correction

src_lines = read_file('/dataset/europarl-inmt/fr-en/test.en')
trg_lines = read_file('/dataset/europarl-inmt/fr-en/test.fr')

file_out = open('/dataset/europarl-inmt/fr-en/sb_t5.fr', 'w')

model_name = "google/mt5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, device_map="auto")
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def translate(c_sentence):
	prompt = "English: {0}\n French:\n".format(c_sentence)

	input_ids = tokenizer(prompt, return_tensors="pt").to(device)
	generated_tokens = model.generate(**input_ids)
	c_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

	return ' '.join(wordpunct_tokenize(c_translation))

def segment_based(c_sentence, c_segments, ref):
	prompt = "English: {0}\n French:\n".format(c_sentence)

	# Anyadir nueva palabra
	correction = 0
	if c_segments==[] or c_segments[0][0]!=0:
		c_segments.insert(0, [0, [ref[0]]])
		correction = 1
	else:
		len_com = len(c_segments[0][1])
		if len_com < len(ref):
			c_segments[0][1].append(ref[len_com])
			correction = 1

	extra_id = 0
	if c_segments!=[]:
		if c_segments[0][0]!=0:
			prompt += "<extra_id_{0}> ".format(extra_id)
			extra_id += 1
		for i in c_segments:
			prompt += ' '.join(i[1]) + " "
			prompt += "<extra_id_{0}> ".format(extra_id)
			extra_id += 1
	#print(c_segments)
	#print(prompt)

	input_ids = tokenizer(prompt, return_tensors="pt").to(device)
	generated_tokens = model.generate(**input_ids)
	c_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
	#print(c_output)
	c_output = c_output.replace('</s>', '')
	elements = re.split('<extra_id_[0-99]*>', c_output)[1:]
	#print(elements)

	c_translation = ""
	n_ele = 0
	if c_segments!=[]:
		if c_segments[0][0]!=0:
			c_translation += elements[n_ele]
			n_ele += 1
		for i in c_segments:
			c_translation += ' '.join(i[1]) + " "
			if n_ele < len(elements):
				c_translation += elements[n_ele] + " "
				n_ele += 1

	return c_translation, correction

total_words = 0
total_ws = 0
total_ma = 0
for i in range(0, len(src_lines)):
	#if i<374:
	#	continue

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
		_, segments = get_segments(hyp, ref)
		segments, correction = merge_segments(segments)
		mouse_actions += correction
		c_translation, correction = segment_based(c_sentence, segments, ref)
		if correction == 1:
			mouse_actions += 1
			word_strokes  += 1

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
