"""
Segment-Based Approach with Mbart

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
import argparse
import sys

import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
from transformers import (MBart50TokenizerFast, MBartForConditionalGeneration,
                          PhrasalConstraint)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
wordTokenizer = TreebankWordTokenizer()

def read_file(name):
	file_r = open(name, 'r')
	lines = file_r.read().splitlines()
	file_r.close()
	return lines

def tokenize(sentence):
	sentence = sentence.replace('…', '...')
	sentence = sentence.replace('´', '\'')
	sentence = sentence.replace('\'', ' \' ')
	sentence = sentence.replace('.', ' . ')
	tokens = wordTokenizer.tokenize(sentence)
	for idx, t in enumerate(tokens):
		t = t.replace('``', '"')
		tokens[idx] = t
	return tokens

def check_prefix(target, hyp):
	prefix = []
	correction = 0

	target = tokenize(target)
	hyp = tokenize(hyp)

	for i in range(len(target)):
		if len(hyp)<=i:
			correction = 1
			prefix.append(target[i])
			break
		elif target[i] == hyp[i]:
			prefix.append(target[i])
		else:
			correction = 1
			prefix.append(target[i])
			if target[i] == hyp[i][:len(target[i])] and len(target)>i+1:
				correction = 2
				prefix.append(target[i+1])
			break
	prefix = ' '.join(prefix)
	prefix += ' '
	return prefix, correction

def check_segments(target,hyp):
	target = tokenize(target)
	hyp = tokenize(hyp)
	segments = []; wrong_words = []; buffer = []
	good_segment = False
	while hyp:
		while target and hyp and target[0] == hyp[0]:
			buffer.append(target[0])
			target = target[1:]
			hyp = hyp[1:]
			good_segment = True
		segments.append(buffer)
		if good_segment and hyp:
			wrong_words.append(hyp[0])
			hyp = hyp[1:]
			good_segment = False
		h = 0
		while target and target[0] != hyp[0]:
			while h < len(hyp) and target[0] != hyp[h]:
				h += 1
			if target[0] == hyp[h]:
				hyp = hyp[h:]
			else:
				target = target[1:]
	return segments, wrong_words

def translate(args):
	try:
		#|========================================================
		#| READ SOURCE AND TARGET DATASET
		file_name = '{0}/test.{1}'.format(args.folder, args.source)
		src_lines = read_file(file_name)
		file_name = '{0}/test.{1}'.format(args.folder, args.target)
		trg_lines = read_file(file_name)
		#| PREPARE DOCUMENT TO WRITE
		file_name = '{0}/imt_mbart.{1}'.format(args.folder, args.target)
		file_out = open(file_name, 'w')
		#|========================================================
		#| LOAD MODEL AND TOKENIZER
		model_path = args.model
		model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
		tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang=args.source_code, tgt_lang=args.target_code)
		#|========================================================
		#| PREPARE PREFIX FORCER
		prefix = []
		vocab = [*range(len(tokenizer))]
		def restrict_prefix(batch_idx, prefix_beam):
			pos = len(prefix_beam)
			if pos<len(prefix):
				return [prefix[pos]]
			ids = vocab
			return ids
		#|========================================================

		total_words = 0
		total_chars = 0
		total_ws = 0
		total_ma = 0
		for i in range(0, len(src_lines)):
			#if i<1280-1:
			#	continue
			# Save the SRC and TRG sentences
			c_src = src_lines[i]
			c_trg = ' '.join(tokenize(trg_lines[i]))

			mouse_actions = 0
			word_strokes = 0
			n_words = len(tokenize(trg_lines[i]))
			n_chars = len(trg_lines[i])

			# Convert them to ids
			encoded_src = tokenizer(c_src, return_tensors="pt").to(device)
			encoded_trg = [2] + tokenizer(text_target=c_trg).input_ids[:-1]

			# Prints
			#print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,c_src,c_trg))

			ite = 0
			prefix = []
			len_old_prefix = 0
			MAX_TOKENS = 128
			while (prefix[:len(encoded_trg)] != encoded_trg):
				# Generate the translation
				generated_tokens = model.generate(**encoded_src,
									forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
									max_new_tokens=MAX_TOKENS,
									prefix_allowed_tokens_fn=restrict_prefix).tolist()[0]
				if len(generated_tokens) >= MAX_TOKENS:
					MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))
				output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
				prefix, correction = check_prefix(c_trg, output)
				prefix = [2] + tokenizer(text_target=prefix).input_ids[:-1]

				
				if correction == 0:
					if len(prefix) != len_old_prefix +1:
						mouse_actions += 1
				elif correction == 1:
					if len(prefix) != len_old_prefix +1:
						mouse_actions += 1
					word_strokes += 1
				elif correction == 2:
					if len(prefix) != len_old_prefix +1:
						mouse_actions += 2
					word_strokes += 2
				len_old_prefix = len(prefix)

				#print("ITE {0}: {1}".format(ite, output))
				ite += 1

				#file_out.write("{}\n".format(output[0]))
			#print("WSR: {0:.4f} MAR: {1:.4f}".format(word_strokes/n_words, mouse_actions/n_chars))
			#print("Total Mouse Actions: {}".format(mouse_actions))
			#print("Total Word Strokes: {}".format(word_strokes))
			total_words += n_words
			total_chars += n_chars
			total_ws += word_strokes
			total_ma += mouse_actions

			if (i+1)%10 == 0:
				output_txt = "Line {0} T_WSR: {1:.4f} T_MAR: {2:.4f}".format(i, total_ws/total_words, total_ma/total_chars)
				print(output_txt)
			#print("\n")
			file_out.write("{2} T_WSR: {0:.4f} T_MAR: {1:.4f}\n".format(total_ws/total_words, total_ma/total_chars, i))
			file_out.flush()
		file_out.close()
	except:
		file_out.write("T_WSR: {0:.4f} T_MAR: {1:.4f}\n".format(total_ws/total_words, total_ma/total_chars))
		file_out.close()

def check_language_code(code):
	if code=='ar':			# Arabic
		return 'ar_AR'
	elif code == 'cs':		# Czech
		return 'cs_CZ'
	elif code == 'de':		# German
		return 'de_DE'
	elif code == 'en':		# English
		return 'en_XX'
	elif code == 'es':		# Spanish
		return 'es_XX'
	elif code == 'et':		# Estonian
		return 'et_EE'
	elif code == 'fi':		# Finnish
		return 'fi_FI'
	elif code == 'fr':		# French
		return 'fr_XX'
	elif code == 'gu':		# Gujarati
		return 'gu_IN'
	elif code == 'hi':		# Hindi
		return 'hi_IN'
	elif code == 'it':		# Italian
		return 'it_IT'
	elif code == 'ja':		# Japanese
		return 'ja_XX'
	elif code == 'kk':		# Kazakh
		return 'kk_KZ'
	elif code == 'ko':		# Korean
		return 'ko_KR'
	elif code == 'lt':		# Lithuanian
		return 'lt_LT'
	elif code == 'lv':		# Latvian
		return 'lv_LV'
	elif code == 'my':		# Burmese
		return 'my_MM'
	elif code == 'ne':		# Nepali
		return 'ne_NP'
	elif code == 'nl':		# Ducht
		return 'nl_XX'
	elif code == 'ro':		# Romanian
		return 'ro_RO'
	elif code == 'ru':		# Russian
		return 'ru_RU'
	elif code == 'si':		# Sinhala
		return 'si_LK'
	elif code == 'tr':		# Turkish
		return 'tr_TR'
	elif code == 'vi':		# Vietnamese
		return 'vi_VN'
	elif code == 'zh':		# Chinese
		return 'zh_CN'
	elif code == 'af':		# Afrikaans
		return 'af_ZA'
	elif code == 'az':		# Azerbaijani
		return 'az_AZ'
	elif code == 'bn':		# Bengali
		return 'bn_IN'
	elif code == 'fa':		# Persian
		return 'fa_IR'
	elif code == 'he':		# Hebrew
		return 'he_IL'
	elif code == 'hr':		# Croatian
		return 'hr_HR'
	elif code == 'id':		# Indonesian
		return 'id_ID'
	elif code == 'ka':		# Georgian
		return 'ka_GE'
	elif code == 'km':		# Khmer
		return 'km_KH'
	elif code == 'mk':		# Macedonian
		return 'mk_MK'
	elif code == 'ml':		# Malayalam
		return 'ml_IN'
	elif code == 'mn':		# Mongolian
		return 'mn_MN'
	elif code == 'mr':		# Marathi
		return 'mr_IN'
	elif code == 'pl':		# Polish
		return 'pl_PL'
	elif code == 'ps':		# Pashto
		return 'ps_AF'
	elif code == 'pt':		# Portuguese
		return 'pt_XX'
	elif code == 'sv':		# Swedish
		return 'sv_SE'
	elif code == 'sw':		# Swahili
		return 'sw_KE'
	elif code == 'ta':		# Tamil
		return 'ta_IN'
	elif code == 'te':		# Telegu
		return 'te_IN'
	elif code == 'th':		# Thai
		return 'th_TH'
	elif code == 'tl':		# Tagalog
		return 'tl_XX'
	elif code == 'uk':		# Ukrainian
		return 'uk_UA'
	elif code == 'ur':		# Urdu
		return 'ur_PK'
	elif code == 'xh':		# Xhosa
		return 'xh_ZA'
	elif code == 'gl':		# Galician
		return 'gl_ES'
	elif code == 'sl':		# Slovene
		return 'sl_SI'
	else:
		print('Code not implemented')
		sys.exit()

def check_parameters(args):
	# Check Source Language
	args.source_code = check_language_code(args.source)

	# Check Target Language
	args.target_code = check_language_code(args.target)

	# Check the model that is going to load
	if args.model == None:
		args.model = "./mbart-large-50-many-to-many-mmt"

	return args

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument("-model", "--model", required=False, help="Model to load")

	args = parser.parse_args()
	return args

def main():
	# Read Parameters
	args = read_parameters()
	args = check_parameters(args)
	print(args)
	
	translate(args)

if __name__ == "__main__":
	main()
