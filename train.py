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
import lightning as L
from datasets import DatasetDict
from evaluate import load
import bitsandbytes as bnb
import numpy as np
import evaluate
import argparse
import torch
import sys
import os

MODEL = None
METRIC = None
TOKENIZER = None
os.environ['TOKENIZERS_PARALLELISM']='true'

class MosesCorpus(torch.utils.data.Dataset):
	def __init__(self,source,target,tok,prefix=''):
		self.src = []
		with open(source,'r') as file:
			self.src = [l for l in file]
		self.src = [prefix + l for l in self.src]
		self.tgt = []
		with open(target,'r') as file:
			self.tgt = [l for l in file]
		self.tgt = [l for l in self.tgt]
		self.inputs = tok(self.src, text_target=self.tgt, max_length=128,truncation=True, padding=True, return_tensors='pt')
    
	def __len__(self):
		return len(self.src)
	
	def __getitem__(self,idx):
		return {'input_ids': self.inputs['input_ids'][idx], 
		  'attention_mask': self.inputs['attention_mask'][idx], 
		  'labels': self.inputs['labels'][idx]}

class TranslationModel(L.LightningModule):
	def __init__(self, model, tokenizer,lr=1e-5):
		super().__init__()
		self.model = model
		self.tokenizer = tokenizer
		self.metric = evaluate.load("sacrebleu")
		self.lr = lr
	
	def forward(self, **inputs):
		return self.model(**inputs)
	
	def training_step(self, batch, batch_idx):
		outputs = self.model(**batch)
		loss = outputs.loss
		self.log('train_loss', loss)
		return loss
	
	def validation_step(self, batch, batch_idx):
		outputs = self.model.generate(**batch, max_new_tokens=128)
		hyp = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
		ref = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
		bleu = self.metric.compute(predictions=hyp, references=ref)
		loss = self.model(**batch).loss
		metrics = {'val_loss': loss, 'val_bleu': bleu['score']}
		self.log_dict(metrics)
		return metrics

	def configure_optimizers(self):
		opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
		return {'optimizer': opt,
                'lr_scheduler': torch.optim.lr_scheduler.LinearLR(opt,start_factor=1, end_factor=1/3, total_iters=10000)}


def load_model(model_name, _dev=None):
	if model_name == 'mbart':
		_mdl = 	MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
	elif model_name == 'm2m':
		_mdl = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
	elif model_name == 'flant5':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
	elif model_name == 'mt5':
		_mdl = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
	elif model_name == 'llama3':
		_mdl = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
	elif model_name == 'nllb':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
	elif model_name == 'bloom':
		_mdl = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m')
	else:
		print('Model not implemented: {0}'.format(model_name))
		sys.exit(1)
	return _mdl


def load_tokenizer(args):
	if args.model_name == 'mbart':
		_tok = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
	elif args.model_name == 'm2m':
		_tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
	elif args.model_name == 'flant5':
		_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
	elif args.model_name == 'mt5':
		_tok = AutoTokenizer.from_pretrained("google/mt5-small")
	elif args.model_name == 'llama3':
		_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
	elif args.model_name == 'nllb':
		_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
	elif args.model_name == 'bloom':
		_tok = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
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


def load_datasets(args):
	if 't5' in args.model_name:
		extend = {'en':'English','fr':'French','de':'German','es':'Spanish', 'gl':'Galician','bn':'Bengali','sw':'swahili'}
		prefix = f'translate from {extend[args.source]} to {extend[args.target]}: '
	else:
		prefix = ''
	shards = [	f"{args.folder}train.{args.source}", 
				f"{args.folder}train.{args.target}"
				]
	training = MosesCorpus(shards[0],shards[1],TOKENIZER,prefix = prefix)

	shards = [	f"{args.folder}dev.{args.source}",
				f"{args.folder}dev.{args.target}"
				]	
	development = MosesCorpus(shards[0],shards[1],TOKENIZER, prefix = prefix)
	return training, development

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
	args.source_code = check_language_code(args.source) if args.model_name == 'mbart' else args.source
	args.target_code = check_language_code(args.target) if args.model_name == 'mbart' else args.target
	return args

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument('-model','--model_name',default='mbart',choices=['mbart','m2m','flant5','mt5','llama3','nllb','bloom'],help='Model to train')
	parser.add_argument('-lora','--lora',action='store_true',help='Whether to use LowRank or not')
	parser.add_argument('-quant','--quantize',action='store_true',help='Whether to quantize the model before training or not')
	parser.add_argument("-e","--epochs",type=int,default=3,help="Number of epochs")
	parser.add_argument('-bs','--batch_size',type=int,default=32,help='Batch size')
	parser.add_argument('-lr','--learning_rate',type=float,default=2e-5,help='Learning rate of the optimizer')

	args = parser.parse_args()
	return args

def main():
	global TOKENIZER, METRIC, MODEL

	args = read_parameters()
	args = check_parameters(args)
	print(args)

	METRIC = load("sacrebleu")
	MODEL = load_model(args.model_name)
	TOKENIZER = load_tokenizer(args)

	if args.lora:
		lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules='all-linear'
    	)
		MODEL = get_peft_model(MODEL, lora_config)

	num_workers = int(os.cpu_count() * 0.75)
	train_dataset, dev_dataset = load_datasets(args)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

	fp16 = not 't5' in args.model_name
	
	translator = TranslationModel(MODEL, TOKENIZER, lr=args.learning_rate)
	callbacks = [L.pytorch.callbacks.EarlyStopping(monitor='val_bleu', mode='max', patience=3, min_delta=1e-5),
                L.pytorch.callbacks.ModelCheckpoint(monitor='val_bleu', mode='max', save_top_k=3, save_weights_only=True,
								  dirpath=f'models/{args.model_name}_{args.source+args.target}')]
	trainer = L.Trainer(max_epochs=args.epochs,
					 precision=16 if fp16 else 32,
					 val_check_interval=0.2,
					 default_root_dir=f'models/{args.model_name}_{args.source+args.target}',
					 callbacks=callbacks)
	
	trainer.validate(model=translator,dataloaders=dev_dataloader)
	trainer.fit(model=translator, train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader)
	trainer.validate(model=translator,dataloaders=dev_dataloader)



if __name__ == '__main__':
	main()
