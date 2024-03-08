"""
Fine Tune Mbart Model

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from datasets import DatasetDict, load_metric
import numpy as np
import argparse
import torch
import sys

MODEL = None
METRIC = None
TOKENIZER = None

class Europarl(torch.utils.data.Dataset):
    def __init__(self,source,target,tok):
        self.src = []
        with open(source,'r') as file:
            self.src = [l for l in file]
        self.tgt = []
        with open(target,'r') as file:
            self.tgt = [l for l in file]
        self.tok = tok
    
    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.tok(self.src[idx], text_target=self.tgt[idx], max_length=128, truncation=True)

def get_device():
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	return device


def load_model(model_path, _dev=None):
	_mdl = MBartForConditionalGeneration.from_pretrained(model_path).to(_dev)
	return _mdl


def load_tokenizer(tokenizer_path, args):
	_tok = MBart50TokenizerFast.from_pretrained(tokenizer_path)
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
	shards = [	f"{args.trn}tr.{args.source}", 
				f"{args.trn}tr.{args.target}"
				]
	training = Europarl(shards[0],shards[1],TOKENIZER)

	shards = [	f"{args.dev}dev.{args.source}",
				f"{args.dev}dev.{args.target}"
				]	
	development = Europarl(shards[0],shards[1],TOKENIZER)

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

def check_parameters(args):
	args.source_code = check_language_code(args.source)
	args.target_code = check_language_code(args.target)
	return args

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")

	args = parser.parse_args()
	return args

def main():
	global TOKENIZER, METRIC, MODEL

	args = read_parameters()
	args = check_parameters(args)
	print(args)

	device = get_device()
	METRIC = load_metric("sacrebleu")
	MODEL = load_model("facebook/mbart-large-50-many-to-many-mmt", device)
	TOKENIZER = load_tokenizer("facebook/mbart-large-50-many-to-many-mmt", args)

	dataset = load_datasets(args)

	training_args = Seq2SeqTrainingArguments(
		'models/europarl_{0}'.format(args.source+args.target),
		evaluation_strategy='steps',
		eval_steps=10000,
		learning_rate=2e-5,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		weight_decay=0.01,
		save_total_limit=10,
		num_train_epochs=3,
		predict_with_generate=True,
		fp16=True,
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
	trainer.train()



if __name__ == '__main__':
	main()

