from transformers import BloomTokenizerFast, BloomForCausalLM, AutoModelForCausalLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
import numpy as np
import evaluate

# Code development following the next tutorial
# https://huggingface.co/docs/transformers/v4.18.0/en/tasks/translation
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb#scrollTo=vc0BSBLIIrJQ

class CustomTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		"""
		How the loss is computed by Trainer. By default, all models return the loss in the first element.
		Subclass and override for custom behavior.
		"""
		if self.label_smoother is not None and "labels" in inputs:
		    labels = inputs.pop("labels")
		else:
		    labels = None
		outputs = model(**inputs)
		# Save past state if it exists
		# TODO: this needs to be fixed and made cleaner later.
		if self.args.past_index >= 0:
		    self._past = outputs[self.args.past_index]
		if labels is not None:
		    if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
		        loss = self.label_smoother(outputs, labels, shift_labels=True)
		    else:
		        loss = self.label_smoother(outputs, labels)
		else:
		    if isinstance(outputs, dict) and "loss" not in outputs:
		        raise ValueError(
		            "The model did not return a loss from the inputs, only the following keys: "
		            f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
		        )
		    # We don't use .loss here since the model may return tuples instead of ModelOutput.
		    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
		return (loss, outputs) if return_outputs else loss	

def preprocess_function(examples):
	prefix = "Translate from english to french:\n"
	inputs  = [prefix + example['en'] + " ::: " for example in examples['translation']]
	targets = [example['fr'] for example in examples['translation']]

	model_inputs = tokenizer(inputs, max_length=64, padding='max_length', truncation=True)
	#model_inputs = tokenizer(inputs)

	with tokenizer.as_target_tokenizer():
		labels = tokenizer(targets, max_length=64, padding='max_length', truncation=True)
		#labels = tokenizer(targets)

	model_inputs['labels'] = labels['input_ids'].copy()
	return model_inputs 

def postprocess_text(preds, labels):
	preds = [pred.strip() for pred in preds]
	labels = [[label.strip()] for label in labels]

	return preds, labels

metric = load_metric("sacrebleu")
def compute_metrics(eval_preds):
	preds, labels = eval_preds
	print(preds)
	print(labels)
	if isinstance(preds, tuple):
		preds = preds[0]
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

	# Replace -100 in the labels as we can't decode them.
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

	# Some simple post-processing
	decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

	result = metric.compute(predictions=decoded_preds, references=decoded_labels)
	result = {"bleu": result["score"]}

	prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
	result["gen_len"] = np.mean(prediction_lens)
	result = {k: round(v, 4) for k, v in result.items()}
	return result

#|===============================================
#| CARGAR MODELO Y TOKENIZADOR
lm_option = 'default_Model'
lm_path = './bloom-560m'
if lm_option == 'bloom_Model':
	model 	  = BloomForCausalLM.from_pretrained(lm_path, device_map="auto")
	tokenizer = BloomTokenizerFast.from_pretrained(lm_path)
elif lm_option == 'default_Model':
	model 	  = AutoModelForCausalLM.from_pretrained(lm_path, device_map="auto")
	tokenizer = AutoTokenizer.from_pretrained(lm_path, use_fast=True)
#|===============================================

# Load a translation dataset
dataset = load_dataset("opus_books", "en-fr")
dataset = dataset['train'].train_test_split(test_size=0.0005)
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['id', 'translation'], desc='Running tokenizer on dataset')
print(tokenized_datasets['train'][1])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Load Hyperparameters
training_args = TrainingArguments(
	output_dir="test_trainer", 
	evaluation_strategy='steps',
	eval_steps=100,
	learning_rate=2e-5,
	per_device_train_batch_size=1,
	per_device_eval_batch_size=1,
	weight_decay=0.01,
	save_total_limit=3,
	num_train_epochs=1,
	eval_accumulation_steps=10,
	)

trainer = CustomTrainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_datasets['train'],
		eval_dataset=tokenized_datasets['test'],
		#tokenizer=tokenizer,
		#compute_metrics=compute_metrics,
		#data_collator=data_collator,
	)

"""
TEST
import torch
inputs_ids = torch.IntTensor([tokenized_datasets['train'][1]['input_ids']])
inputs_ids = inputs_ids.to('cuda')
print(model.generate(inputs_ids, max_new_tokens=100))
"""
print('TRAINING')
print(tokenized_datasets["train"][1])
print('TEST')
print(tokenized_datasets["test"][1])
print(type(tokenized_datasets['train'][1]['input_ids'][24]))
trainer.train()

