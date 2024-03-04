"""
Fine Tune Mbart Model

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_metric
import numpy as np
import argparse
import torch
import sys

MODEL = None
METRIC = None
TOKENIZER = None

def get_device():
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	return device


def load_model(model_path, _dev=None):
	_mdl = MT5ForConditionalGeneration.from_pretrained(model_path).to(_dev)
	return _mdl


def load_tokenizer(tokenizer_path, args):
	_tok = AutoTokenizer.from_pretrained(tokenizer_path)
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
	shards = [	f"{args.trn}.{args.source}", 
				f"{args.trn}.{args.target}"
				]
	raw_training = Dataset.from_generator(gen, gen_kwargs={"shards": shards})
	tokenized_training = raw_training.map(preprocess_dataset, batched=True)


	shards = [	f"{args.dev}.{args.source}",
				f"{args.dev}.{args.target}"
				]
	raw_development = Dataset.from_generator(gen, gen_kwargs={"shards": shards})
	tokenized_development = raw_development.map(preprocess_dataset, batched=True)	


	tokenized_dataset = DatasetDict({"train":tokenized_training,"test":tokenized_development})
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

def check_parameters(args):
	return args

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-trn", "--trn", required=True, help="Folder where is the training dataset")
	parser.add_argument("-dev", "--dev", required=True, help="Folder where is the development dataset")

	args = parser.parse_args()
	return args

def main():
	global TOKENIZER, METRIC, MODEL

	args = read_parameters()
	args = check_parameters(args)
	print(args)

	device = get_device()
	print(device)
	METRIC = load_metric("sacrebleu")
	MODEL = load_model("google/mt5-base", device)
	TOKENIZER = load_tokenizer("google/mt5-base", args)

	dataset = load_datasets(args)

	training_args = Seq2SeqTrainingArguments(
		'models/europarl_{0}'.format(args.source+args.target),
		#evaluation_strategy='steps',
		#eval_steps=100000000000000000000000,
		save_steps=10000,
		save_total_limit = 30,
		learning_rate=2e-5,
		per_device_train_batch_size=64,
		per_device_eval_batch_size=64,
		weight_decay=0.01,
		num_train_epochs=5,
		predict_with_generate=True,
		fp16=False, # Tiene que estar a False para que funcione con MT5
		#load_best_model_at_end=True,
		#metric_for_best_model = 'bleu'
		)

	data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=MODEL)


	trainer = Seq2SeqTrainer(
		MODEL,
		training_args,
		train_dataset=dataset['train'],
		eval_dataset=dataset['test'],
		data_collator=data_collator,
		tokenizer=TOKENIZER,
		#compute_metrics=compute_metrics,
		#callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
		)
	trainer.train()



if __name__ == '__main__':
	main()

