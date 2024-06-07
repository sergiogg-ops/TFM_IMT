import argparse
import sys

import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
import evaluate
import bitsandbytes as bnb
from tqdm import tqdm
from transformers import (MBart50TokenizerFast, MBartForConditionalGeneration,
						  M2M100ForConditionalGeneration, M2M100Tokenizer,
						  AutoModelForSeq2SeqLM, AutoTokenizer,
                          TranslationPipeline, MT5ForConditionalGeneration)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
wordTokenizer = TreebankWordTokenizer()

class TranslationPipelineWithProgress:
    def __init__(self, model,tokenizer, batch_size, device):
        self.translator = TranslationPipeline(model=model,tokenizer=tokenizer, batch_size=batch_size, device=device)
        self.batch_size = batch_size

    def __call__(self, texts, **kwargs):
        translations = []
        # Process texts in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Traduciendo", unit="batch"):
            batch_texts = texts[i:i + self.batch_size]
            batch_translations = self.translator(batch_texts, **kwargs)
            translations.extend([t['translation_text'] for t in batch_translations])
        return translations

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

def load_model(model_path, args, _dev=None):
	if args.model_name == 'mbart':
		_mdl = MBartForConditionalGeneration.from_pretrained(model_path,attn_implementation="flash_attention_2").to(_dev)
		_tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'm2m':
		_mdl = M2M100ForConditionalGeneration.from_pretrained(model_path).to(_dev)
		_tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'flant5':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(_dev)
		_tok = AutoTokenizer.from_pretrained("google/flan-t5-base",src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'mt5':
		_mdl = MT5ForConditionalGeneration.from_pretrained(model_path).to(_dev)
		_tok = AutoTokenizer.from_pretrained("google/mt5-small")
	else:
		print('Model not implemented: {0}'.format(args.model_name))
		sys.exit(1)
	return _mdl, _tok

def translate(args):
	print('Cargando modelo...')
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.source)
	src_lines = read_file(file_name)
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.target)
	trg_lines = read_file(file_name)
	if 't5' in args.model_name:
		extend = {'en':'English','fr':'French','de':'German','es':'Spanish'}
		prefix = f'translate from {extend[args.source]} to {extend[args.target]}: '
		src_lines = [prefix + l for l in src_lines]
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model, tokenizer = load_model(model_path, args, device)
	#|========================================================
	MAX_TOKENS = 400
	bleu_metric = evaluate.load('bleu',trust_remote_code=True)
	ter_metric = evaluate.load('ter',trust_remote_code=True)
	translator = TranslationPipelineWithProgress(model=model,tokenizer=tokenizer, batch_size=args.batch_size, device=device)
	#|========================================================
	#| TRANSLATE
	print('Traduciendo...')
	translation = translator(src_lines, src_lang=args.source_code, tgt_lang=args.target_code, max_length=MAX_TOKENS)
	hypothesis = [t['translation_text'] for t in translation]
	print(type(translation))
	print('Evaluando metricas...')
	bleu = bleu_metric.compute(predictions=hypothesis,references=trg_lines)
	ter = ter_metric.compute(predictions=hypothesis,references=trg_lines)
	print('BLEU:')
	print(f'\t{bleu}')
	print('TER:')
	print(f'\t{ter}')
		
	

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
	parser.add_argument("-p","--partition", required=False, default="test", choices=["dev","test"], help="Partition to load")
	parser.add_argument("-model", "--model", required=False, help="Model to load")
	parser.add_argument("-model_name", "--model_name", required=False, choices=['mbart','m2m','flant5','mt5'], help="Model to load")
	parser.add_argument('-b','--batch_size',required=False,default=64,type=int,help='Batch size for the inference')
	parser.add_argument('-quant','--quantize',action='store_true',help='Whether to quantize the model or not')

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
