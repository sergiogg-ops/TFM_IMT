"""
Segment-Based Approach with T5
Testing with English-French
"""

from transformers import T5TokenizerFast, T5Tokenizer, MT5ForConditionalGeneration, T5ForConditionalGeneration
from nltk.tokenize.treebank import TreebankWordTokenizer
from mosestokenizer import MosesTokenizer
from nltk.tokenize import wordpunct_tokenize
import argparse
import re

import constraints
from constraints import Segmentor, MT5_Fachade, MBART_Fachade

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
	any_correction = 0

	for i in c_segments:
		if segments == []:
			segments.append(i)
		else:
			l_segment = segments[-1]

			if l_segment[0]+len(l_segment[1])==i[0]:
				any_correction += 1
				l_segment[1] += i[1]
			else:
				segments.append(i)

	return segments, any_correction

def segment_based(MODEL, src, c_segments, ref, hyp):
	# Anyadir nueva palabra
	word_correction = None
	if c_segments==[] or c_segments[0][0]!=0:
		c_segments.insert(0, [0, [ref[0]]])
		word_correction = ref[0]
	else:
		len_com = len(c_segments[0][1])
		if len_com < len(ref):
			c_segments[0][1].append(ref[len_com])
			word_correction = ref[len_com]

	c_segments = [ ' '.join(segment[1]) for segment in c_segments]
	translation = MODEL.segment_translate(src, c_segments)

	return translation, word_correction

def interactive_session(args):
	# MBART_Fachade / MT5_Fachade
    MODEL = MT5_Fachade(args.model, args.source, args.target)

    SRC_DATA = read_file(f"{args.folder}/test.{args.source}")
    TRG_DATA = read_file(f"{args.folder}/test.{args.target}")
    FILE_OUT = open(f"{args.folder}/sb_t5.{args.target}", 'w')

    total_n_words = 0
    total_n_chars = 0
    total_word_strokes = 0
    total_char_strokes = 0
    total_mouse_actions = 0
    for idx, (src_line, trg_line) in enumerate(zip(SRC_DATA, TRG_DATA)):
        #if idx<1316:
        #    continue

        n_iteration = 0
        reference  = MODEL.tokenize(trg_line, args.target)
        print(f"\nProcessing sentence {idx}: {src_line}\nReference: {' '.join(reference)}")

        token_size = len(MODEL.encode(' '.join(reference)))
        if  token_size > 190:
            print(f"\nWARNING")
            print(f"The size of the sentence {idx} is {token_size} tokens. Bigger than the 200 max.")
            continue

        hypothesis = MODEL.translate(src_line)
        print(f"\tITE {n_iteration}: {' '.join(hypothesis)}")
        n_iteration += 1

        mouse_actions = 0
        word_strokes = 0
        char_strokes = 0

        number_words = len(reference)
        number_chars = len(''.join(reference))
        while (hypothesis[:number_words] != reference):
            _, segments = get_segments(hypothesis, reference)
            segments, any_correction = merge_segments(segments)
            mouse_actions += any_correction
	    
            hypothesis, word_correction = segment_based(MODEL, src_line, segments, reference, hypothesis)

            if word_correction != None:
                mouse_actions += 1
                word_strokes  += 1
                char_strokes  += len(word_correction)

            print(f"\tITE {n_iteration}: {' '.join(hypothesis)}")
            n_iteration += 1
            

        print("WSR: {0:.4f} MAR: {1:.4f} CSR: {2:.4f}".format(word_strokes/number_words, mouse_actions/number_chars, char_strokes/number_chars))
        total_n_words += number_words
        total_n_chars += number_chars
        total_word_strokes += word_strokes
        total_char_strokes += char_strokes
        total_mouse_actions += mouse_actions
        if (idx+1)%10 == 0:
                print("T_WSR: {0:.4f} T_MAR: {1:.4f} T_CSR: {2:.4f}".format(total_word_strokes/total_n_words, total_mouse_actions/total_n_chars, total_char_strokes/total_n_chars))
                FILE_OUT.write("{3} T_WSR: {0:.4f} T_MAR: {1:.4f} T_CSR: {2:.4f}\n".format(total_word_strokes/total_n_words, total_mouse_actions/total_n_chars, total_char_strokes/total_n_chars, idx))
    FILE_OUT.write("END T_WSR: {0:.4f} T_MAR: {1:.4f} T_CSR: {2:.4f}\n".format(total_word_strokes/total_n_words, total_mouse_actions/total_n_chars, total_char_strokes/total_n_chars))
    FILE_OUT.close()

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument("-model", "--model", required=False, help="Model to perform the segment based session")

	args = parser.parse_args()
	return args

def check_parameters(args):
	if not args.model:
		args.model = "google/mt5-small"
	print(args)
	return args

def main():
    args = read_parameters()
    args = check_parameters(args)

    interactive_session(args)

if __name__ == '__main__':
	main()