import sys
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          M2M100ForConditionalGeneration, M2M100Tokenizer,
                          MBart50TokenizerFast, MBartForConditionalGeneration,
						  AutoModelForCausalLM, AutoModelForImageTextToText,
						  AutoProcessor, AutoModelForImageTextToText)
from warnings import warn
from os import getenv

try:
	HF_TOKEN = getenv('HF_TOKEN')
except Exception:
	HF_TOKEN = None
	raise warn(".env file not found. Create it from .env.example if you want to access to restricted repositories.")
PROMPTERS = ['llama','qwen','eurollm','gemma']
NAMES = ['mbart','m2m','flant5','nllb','llama','qwen','eurollm','gemma']
ISO = {'en':'English',
	   	'ca':'Catalan',
		'fr':'French',
		'de':'German',
		'es':'Spanish',
		'gl':'Galician',
		'bn':'Bengali',
		'sw':'Swahili'}

class Prompter:
	def __init__(self, instruction = ''):
		self. instr = instruction

	def src_format(self, text, src_lang, tgt_lang):
		return text
	
	def tgt_format(self, src, tgt):
		return tgt
	
	def decode(self, tokens, tokenizer, **kwargs):
		return tokenizer.decode(tokens, skip_special_tokens=True, **kwargs)
	
	def batch_decode(self, tokens, tokenizer, **kwargs):
		return tokenizer.batch_decode(tokens, skip_special_tokens=True, **kwargs)
	
	def clean(self, text):
		return text

class LlamaPrompter(Prompter):
	def src_format(self, text):
		return f'<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>{self.instr}<|eot_id|><|start_header_id|>user<|end_header_id|>{text}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>Translation:'
	def tgt_format(self, src, tgt):
		return self.src_format(src) + tgt + '<|eot_id|>'
	def clean(self, text):
		text = text.split('<|start_header_id|>assistant<|end_header_id|>Translation:')[-1].strip()
		return text[:text.find('<|eot_id|>')].strip()
	
class EuroPrompter(Prompter):
	def src_format(self, text, src_lang, tgt_lang):
		src_lang = ISO.get(src_lang, src_lang)
		tgt_lang = ISO.get(tgt_lang, tgt_lang)
		return f'{src_lang}: {text}. {tgt_lang}:'
	def tgt_format(self, src, tgt, src_lang, tgt_lang):
		src_lang = ISO.get(src_lang, src_lang)
		tgt_lang = ISO.get(tgt_lang, tgt_lang)
		return self.src_format(src, src_lang, tgt_lang) + tgt
	def clean(self, src, hyp):
		return hyp[len(src):]

class GemmaPrompter(Prompter):
	def src_format(self, text, src_lang, tgt_lang):
		src_lang = ISO.get(src_lang, src_lang)
		tgt_lang = ISO.get(tgt_lang, tgt_lang)
		return f"<start_of_turn>user\nTranslate the following {src_lang} text to {tgt_lang} without further explanation: {text}<end_of_turn>\n<start_of_turn>model\n"
	def tgt_format(self, src, tgt, src_lang, tgt_lang):
		src_lang = ISO.get(src_lang, src_lang)
		tgt_lang = ISO.get(tgt_lang, tgt_lang)
		return f"<start_of_turn>user\nTranslate the following {src_lang} text to {tgt_lang} without further explanation: {src}<end_of_turn>\n<start_of_turn>model\n{tgt}<end_of_turn>"
	def decode(self, tokens, tokenizer, **kwargs):
		return tokenizer.decode(tokens, skip_special_tokens=False, **kwargs)
	def batch_decode(self, tokens, tokenizer, **kwargs):
		return tokenizer.batch_decode(tokens, skip_special_tokens=False, **kwargs)
	def clean(self, src, hyp):
		text = hyp[len(src):]
		text = text.split('<start_of_turn>model')[-1].strip()
		text = text.split('<end_of_turn>')[0].strip()
		return text

class MosesCorpus(Dataset):
	'''
	Pytorch dataset from a moses format corpus
	'''
	def __init__(self,source,target,prompter = Prompter()):
		'''
		Parameters:
			source (str): Path to the source file
			target (str): Path to the target file
			tok (Tokenizer): Tokenizer to use
			prompt (str): prompt to add to the source text
		'''
		self.src = []
		self.tgt = []
		self.raw_tgt = []
		src_lang = source.split('.')[-1]
		tgt_lang = target.split('.')[-1]
		with open(source,'r') as src_file:
			with open(target,'r') as tgt_file:
				for s, t in zip(src_file, tgt_file):
					self.src.append(prompter.src_format(s, src_lang, tgt_lang))
					self.tgt.append(prompter.tgt_format(s,t, src_lang, tgt_lang))
					self.raw_tgt.append(t.strip())
    
	def __len__(self):
		return len(self.src)
	
	def __getitem__(self,idx):
		# return {'input_ids': self.inputs['input_ids'][idx], 
		#   'attention_mask': self.inputs['attention_mask'][idx], 
		#   'labels': self.inputs['labels'][idx]}
		return {'source': self.src[idx],
		  	'target': self.tgt[idx],
        	'raw_target': self.raw_tgt[idx]}
	
def load_model(model_path, args, _dev='cpu'):
	'''
	Downloads the model and tokenizer.

	Parameters:
		model_path (str): Path of the model.
		args (argparse.Namespace): Arguments of the execution.
		_dev (str): Device to use.
	
	Returns:
		tuple: Model and tokenizer.
	'''
	kwargs = {}
	# if args.quantize:
	# 	kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True,device=_dev)
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
	elif args.model_name == 'nllb':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
											src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'llama':
		_mdl = AutoModelForCausalLM.from_pretrained(model_path, token='HF_TOKEN', **kwargs)
		_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",token='HF_TOKEN', padding_side='left')
		_tok.padding_side = 'left'
	elif args.model_name == 'qwen':
		_mdl = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
		_tok.padding_side = 'left'
	elif args.model_name == 'eurollm':
		_mdl = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("utter-project/EuroLLM-1.7B")
		_tok.padding_side = 'left'
	elif args.model_name == 'gemma':
		_mdl = AutoModelForCausalLM.from_pretrained(model_path, token='HF_TOKEN')
		_tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", token='HF_TOKEN')
		_tok.padding_side = 'left'
	else:
		print('Model not implemented: {0}'.format(args.model_name))
		sys.exit(1)
	# if not args.quantize:
	# 	_mdl.to(_dev)
	_mdl = _mdl.to(_dev)

	if _tok.pad_token is None:
		_tok.pad_token = _tok.eos_token
		_tok.padding_side = 'left'
		_mdl.config.pad_token_id = _mdl.config.eos_token_id
	return _mdl, _tok

def apply_lora(model):
	lora_config = LoraConfig(
		r=16,
		lora_alpha=16,
		lora_dropout=0.1,
		target_modules='all-linear'
	)
	return get_peft_model(model, lora_config)

def load_data(folder,source, target, partition):
	with open(f'{folder}/{partition}.{source}','r') as src_file:
		src_lines = [l for l in src_file]
	with open(f'{folder}/{partition}.{target}','r') as tgt_file:
		tgt_lines = [l for l in tgt_file]
	return src_lines, tgt_lines

def get_prompter(model_name, source, target):
	extend = {'en':'English','ca':'Catalan','fr':'French','de':'German','es':'Spanish', 'gl':'Galician','bn':'Bengali','sw':'Swahili'}
	#prompt = f'Translate the sentence from {extend[source]} to {extend[target]} without further explanation.'
	if model_name == 'flant5' or model_name == 'llama':
		prompter = LlamaPrompter()
	elif model_name == 'eurollm':
		prompter = EuroPrompter()
	elif model_name == 'gemma':
		prompter = GemmaPrompter()
	else:
		prompter = Prompter('')
	#prompter = Prompter(prompt)
	return prompter

def check_language_code(code):
	'''
	Adapts the language code from ISO 639 to the format of mBART.

	Parameters:
		code (str): Language code as in ISO 639.
	
	Returns:
		str: Language code as in mBART.
	'''
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