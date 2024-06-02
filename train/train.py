"""
Fine Tune Mbart Model

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
from transformers import (MBartForConditionalGeneration, MBart50TokenizerFast,
						M2M100ForConditionalGeneration, M2M100Tokenizer,
						AutoTokenizer, AutoModelForSeq2SeqLM,
						AutoModelForCausalLM, MT5ForConditionalGeneration,
						Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict
from evaluate import load
import bitsandbytes as bnb
import numpy as np
import argparse
import torch
import sys

MODEL = None
METRIC = None
TOKENIZER = None

class Europarl(torch.utils.data.Dataset):
	def __init__(self,source,target,tok,prefix=''):
		self.src = []
		with open(source,'r') as file:
			self.src = [l for l in file]
		self.src = [prefix + l for l in self.src]
		self.tgt = []
		with open(target,'r') as file:
			self.tgt = [l for l in file]
		self.tgt = [prefix + l for l in self.tgt]
		self.tok = tok
    
	def __len__(self):
		return len(self.src)
	
	def __getitem__(self,idx):
		return self.tok(self.src[idx], text_target=self.tgt[idx], max_length=128, truncation=True)

def get_device():
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	return device


def load_model(model_name, _dev=None):
	if model_name == 'mbart':
		_mdl = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',attn_implementation="flash_attention_2").to(_dev)
	elif model_name == 'm2m':
		_mdl = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(_dev)
	elif model_name == 'flant5':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small",attn_implementation="flash_attention_2")
		#_mdl = AutoModelForSeq2SeqLM.from_pretrained("models/flant5_enfr/checkpoint-180000")
	elif model_name == 'mt5':
		#_mdl = MT5ForConditionalGeneration.from_pretrained("google/mt5-small",attn_implementation="flash_attention_2")
		_mdl = MT5ForConditionalGeneration.from_pretrained("models/",attn_implementation="flash_attention_2")
	elif model_name == 'llama3':
		_mdl = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",attn_implementation="flash_attention_2")
	else:
		print('Model not implemented: {0}'.format(model_name))
		sys.exit(1)
	return _mdl


def load_tokenizer(args):
	if args.model_name == 'mbart':
		_tok = MBart50TokenizerFast.from_pretrained(f'acebook/mbart-large-50-many-to-many-mmt')
	elif args.model_name == 'm2m':
		_tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
	elif args.model_name == 'flant5':
		_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
	elif args.model_name == 'mt5':
		_tok = AutoTokenizer.from_pretrained("google/mt5-small")
	elif args.model_name == 'llama3':
		_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
	else:
		print('Model not implemented: {0}'.format(args.model_name))
		sys.exit(1)
	_tok.src_lang = args.source_code
	_tok.tgt_lang = args.target_code
	return _tok


def load_text(file_path):
	with open(file_path) as file:
		data = file.read().splitlines()
	return data


def preprocess_dataset(dataset):
	inputs =  [translation['src'] for translation in dataset['translation']]
	targets = [translation['tgt'] for translation in dataset['translation']]
	model_inputs = TOKENIZER(inputs, text_target=targets, max_length=128, truncation=True)

	return model_inputs


def gen(shards):
	src_data = load_text(shards[0])
	tgt_data = load_text(shards[1])

	for src_line, tgt_line in zip(src_data, tgt_data):
		yield {"translation": {'src': src_line, 'tgt': tgt_line}}


def load_datasets(args):
	if 't5' in args.model_name:
		extend = {'en':'English','fr':'French','de':'German','es':'Spanish'}
		prefix = f'translate from {extend[args.source]} to {extend[args.target]}: '
	else:
		prefix = ''
	shards = [	f"{args.folder}tr.{args.source}", 
				f"{args.folder}tr.{args.target}"
				]
	training = Europarl(shards[0],shards[1],TOKENIZER,prefix = prefix)

	shards = [	f"{args.folder}dev.{args.source}",
				f"{args.folder}dev.{args.target}"
				]	
	development = Europarl(shards[0],shards[1],TOKENIZER,src_lang=args.source,tgt_lang=args.target)

	tokenized_dataset = DatasetDict({"train":training,"test":development})
	return tokenized_dataset


def postprocess_text(preds, labels):
	preds  = [pred.strip() for pred in preds]
	labels = [[label.strip()] for label in labels]

	return preds, labels

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	if isinstance(preds, tuple):
		preds = preds[0]
	decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)

	labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
	decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens = True)

	decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

	result = METRIC.compute(predictions=decoded_preds, references=decoded_labels)
	result = {"bleu": result["score"]}

	prediction_lens = [np.count_nonzero(pred != TOKENIZER.pad_token_id) for pred in preds]
	result["gen_len"] = np.mean(prediction_lens)
	result = {k: round(v,4) for k, v in result.items()}
	return result

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

def quantize_model(model):
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Linear):
			quantized_module = bnb.nn.Linear8bitLt(
				module.in_features, module.out_features, bias=module.bias is not None
			)
			quantized_module.weight.data = module.weight.data
			if module.bias is not None:
				quantized_module.bias.data = module.bias.data
			setattr(model, name, quantized_module)
	return model

def check_parameters(args):
	args.source_code = check_language_code(args.source) if args.model_name == 'mbart' else args.source
	args.target_code = check_language_code(args.target) if args.model_name == 'mbart' else args.target
	return args

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument('-model','--model_name',default='mbart',choices=['mbart','m2m','flant5','mt5','llama3'],help='Model to train')
	parser.add_argument('-lora','--lora',action='store_true',help='Whether to use LowRank or not')
	parser.add_argument('-quant','--quantize',action='store_true',help='Whether to quantize the model before training or not')
	parser.add_argument("-e","--epochs",type=int,default=3,help="Number of epochs")
	parser.add_argument('-bs','--batch_size',type=int,default=32,help='Batch size')

	args = parser.parse_args()
	return args

def main():
	global TOKENIZER, METRIC, MODEL

	args = read_parameters()
	args = check_parameters(args)
	print(args)

	device = get_device()
	METRIC = load("sacrebleu")
	MODEL = load_model(args.model_name, device)
	TOKENIZER = load_tokenizer(args)

	if args.lora:
		lora_config = LoraConfig(
			r=16,
			lora_alpha=16,
			lora_dropout=0.1,
		)
		MODEL = get_peft_model(MODEL, lora_config)
	if args.quantize:
		MODEL = quantize_model(MODEL)

	dataset = load_datasets(args)

	fp16 = not 't5' in args.model_name

	training_args = Seq2SeqTrainingArguments(
		'models/{0}_{1}'.format(args.model_name,args.source+args.target),
		#evaluation_strategy='steps',
		#eval_steps=10000,
		learning_rate=2e-5,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		weight_decay=0.01,
		save_total_limit=3,
		save_steps=10000,
		num_train_epochs=args.epochs,
		predict_with_generate=True,
		fp16=fp16,
		)

	data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=MODEL)


	trainer = Seq2SeqTrainer(
		MODEL,
		training_args,
		train_dataset=dataset['train'],
		eval_dataset=dataset['test'],
		data_collator=data_collator,
		tokenizer=TOKENIZER,
		compute_metrics=compute_metrics
		)
	results = trainer.evaluate()
	print(f'Antes de fine-tunning:\n\tLoss = {results['eval_loss']:.4}\n\tBLEU = {results['eval_bleu']}')
	trainer.train()
	results = trainer.evaluate()
	print(f'Despues de fine-tunning:\n\tLoss = {results['eval_loss']:.4}\n\tBLEU = {results['eval_bleu']}')



if __name__ == '__main__':
	main()

