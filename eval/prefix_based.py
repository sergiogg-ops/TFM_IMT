"""
Segment-Based Approach with Mbart

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
import argparse
import sys
from time import time

import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          M2M100ForConditionalGeneration, M2M100Tokenizer,
                          MBart50TokenizerFast, MBartForConditionalGeneration,
						  MT5ForConditionalGeneration,
						  BitsAndBytesConfig)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
wordTokenizer = TreebankWordTokenizer()
#np.set_printoptions(threshold=np.inf)

class Restrictor():
	def __init__(self,vocab,tokenizer):
		self.vocab = vocab
		self.start_char = '▁'
		self.start_toks = [value for key,value in tokenizer.get_vocab().items() if key[0] == self.start_char]
		self.tokenizer = tokenizer
		self.prefix = []

	def check_segments(self, target, hyp):
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
		self.prefix = self.tokenizer.encode(prefix)[:-1]
		return correction, prefix
	
	def restrict(self,batch_idx, prefix_beam):
		pos = len(prefix_beam)
		if pos<len(self.prefix):
			return [self.prefix[pos]]
		elif pos==len(self.prefix):
			return self.start_toks
		return self.vocab

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
	sentence = sentence.replace(',', ' , ')
	sentence = sentence.replace('-', ' - ')
	tokens = wordTokenizer.tokenize(sentence)
	for idx, t in enumerate(tokens):
		t = t.replace('``', '"')
		tokens[idx] = t
	return tokens

def load_model(model_path, args, _dev=None):
	kwargs = {}
	if args.quantize:
		kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True,device=_dev)
	if args.model_name == 'mbart':
		_mdl = MBartForConditionalGeneration.from_pretrained(model_path, **kwargs)
		_tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", 
											  src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'm2m':
		_mdl = M2M100ForConditionalGeneration.from_pretrained(model_path, **kwargs)
		_tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", 
										 src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'flant5':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("google/flan-t5-base",
									   src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'mt5':
		_mdl = MT5ForConditionalGeneration.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("google/mt5-small",
									   src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'nllb':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
											src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'bloom':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path)
		_tok = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
	else:
		print('Model not implemented: {0}'.format(args.model_name))
		sys.exit(1)
	if not args.quantize:
		_mdl.to(_dev)
	return _mdl, _tok
	
def translate(args):
	#try:
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.source)
	if args.final > -1:
		src_lines = read_file(file_name)[:args.final]
	else:
		src_lines = read_file(file_name)
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.target)
	if args.final > -1:
		trg_lines = read_file(file_name)[:args.final]
	else:
		trg_lines = read_file(file_name)

	if 't5' in args.model_name or args.model_name == 'bloom':
		extend = {'en':'English','fr':'French','de':'German','es':'Spanish'}
		prompt = f'Translate the following sentence from {extend[args.source]} to {extend[args.target]}: '
		src_lines = [prompt + l for l in src_lines]

	#| PREPARE DOCUMENT TO WRITE
	if args.output:
		file_name = '{0}/{1}.{2}'.format(args.folder,args.output, args.target)
	else:
		file_name = '{0}/pb_imt_{1}.{2}'.format(args.folder, args.model_name, args.target)
	file_out = open(file_name, 'w')
	file_out.write(str(args))
	file_out.write("\n")
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model, tokenizer = load_model(model_path, args, device)
	#|=========================================================
	#| PREPARE THE RESTRICTOR
	VOCAB = [*range(len(tokenizer))]
	tiempo_total = 0
	iteraciones = 0
	
	#|=========================================================
	#| GET IN THE RIGHT PLACE

	total_words = 0
	total_chars = 0
	for line in trg_lines[:args.initial]:
		total_words += len(tokenize(line))
		total_chars += len(line)
	total_ws = total_words * args.word_stroke
	total_ma = total_chars * args.mouse_action
	#|=========================================================s	
	for i in range(args.initial, len(src_lines)):
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
		if args.verbose:
			print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,c_src,c_trg))

		ite = 0
		MAX_TOKENS = 256
		restrictor = Restrictor(VOCAB, tokenizer)
		prefix = []
		len_old_prefix = 0
		while prefix[:len(encoded_trg)] != encoded_trg:
			# Generate the translation
			ite += 1

			ini = time()
			raw_output = model.generate(**encoded_src,
							#forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
							max_new_tokens=MAX_TOKENS,
							prefix_allowed_tokens_fn=restrictor.restrict)
			generated_tokens = raw_output.tolist()[0]
			output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
			tiempo_total += time() - ini
			iteraciones += 1
			if len(generated_tokens) >= MAX_TOKENS:
				MAX_TOKENS = min(512, int(MAX_TOKENS*(3/2)))
			elif len(generated_tokens) > 3/4 * MAX_TOKENS:
				MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))

			if args.verbose:
				#print('ITE TOK:', generated_tokens)
				print("ITE {0} ({1}): {2}".format(ite, len(generated_tokens), output))

			correction, prefix = restrictor.check_segments(c_trg, output)
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

			#file_out.write("{}\n".format(output[0]))
		total_words += n_words
		total_chars += n_chars
		total_ws += word_strokes
		total_ma += mouse_actions
		
		output_txt = "Line {0} T_WSR: {1:.4f} T_MAR: {2:.4f} TIME: {3:4f}".format(i, total_ws/total_words, total_ma/total_chars, tiempo_total)
		if args.verbose:
			print(output_txt)
			print("\n")
		file_out.write(f'{word_strokes/n_words}\t{mouse_actions/n_chars}\n')
		file_out.flush()
	output_txt = f"TOTAL => WSR: {total_ws/total_words} - MAR: {total_ma/total_chars} - TIME: {tiempo_total/iteraciones}\n"
	file_out.write(output_txt)
	file_out.close()
	print(output_txt)

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
	args.source_code = check_language_code(args.source) if args.model_name == 'mbart' else args.source

	# Check Target Language
	args.target_code = check_language_code(args.target) if args.model_name == 'mbart' else args.target

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
	parser.add_argument("-out", "--output", required=False, help="Output file")
	parser.add_argument('-model_name','--model_name', required=False, default='mbart', choices=['mbart','m2m','flant5','nllb','bloom'], help='Model name')
	parser.add_argument('-p','--partition',required=False, default='test', choices=['dev','test'], help='Partition to evaluate')
	parser.add_argument("-ini","--initial", required=False, default=0, type=int, help="Initial line")
	parser.add_argument("-fin","--final",required=False, default=-1,type=int,help="Final Line")
	parser.add_argument("-wsr","--word_stroke", required=False, default=0, type=float, help="Last word stroke ratio")
	parser.add_argument("-mar","--mouse_action", required=False, default=0, type=float, help="Last mouse action ratio")
	parser.add_argument('-quant','--quantize',action='store_true',help='Whether to quantize the model or not')
	parser.add_argument("-v","--verbose", required=False, default=False, action='store_true', help="Verbose mode")

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
