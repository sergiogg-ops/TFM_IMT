from model import *
import lightning as L
from evaluate import load
import evaluate
import argparse
import torch
import sys
import os

MODEL = None
METRIC = None
TOKENIZER = None
os.environ['TOKENIZERS_PARALLELISM']='true'

class TranslationModel(L.LightningModule):
	'''
	Pytorch Lightning wrapper module for the training
	'''
	def __init__(self, model, tokenizer,lr=1e-5):
		'''
		Paremeters:
			model (torch.nn.Module): Model to train
			tokenizer (Tokenizer): Tokenizer to use
			lr (float): Learning rate of the optimizer
		'''
		super().__init__()
		self.model = model
		self.tokenizer = tokenizer
		self.metric = evaluate.load("sacrebleu")
		self.lr = lr
	
	def forward(self, inputs):
		input_ids = inputs['input_ids']
		labels = inputs['labels']
		dif = input_ids.shape[-1] - labels.shape[-1]
		if dif > 0:
			labels = torch.nn.functional.pad(labels, (0,dif), value=self.tokenizer.pad_token_id)
		elif dif < 0:
			input_ids = torch.nn.functional.pad(input_ids, (0,-dif), value=self.tokenizer.pad_token_id)
			inputs['attention_mask'] = torch.nn.functional.pad(inputs['attention_mask'], (0,-dif), value=0)
		inputs['input_ids'] = input_ids
		inputs['labels'] = labels
		return self.model(**inputs)
	
	def training_step(self, batch, batch_idx):
		inputs = self.tokenizer(batch['source'], padding=True, text_target=batch['target'], return_tensors='pt').to('cuda')
		outputs = self.forward(inputs)
		loss = outputs.loss
		metrics = {'train_loss': loss}
		self.log_dict(metrics,batch_size=len(batch['source']))
		return loss
	
	def validation_step(self, batch, batch_idx):
		inputs = self.tokenizer(batch['source'], padding=True, text_target=batch['target'], return_tensors='pt').to('cuda')
		# # outputs = self.model.generate(**inputs, max_new_tokens=128)
		# # hyp = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
		# # ref = batch['target']
		# # bleu = self.metric.compute(predictions=hyp, references=ref)
		loss = self.forward(inputs).loss
		metrics = {'val_loss': loss}
		self.log_dict(metrics,batch_size=len(batch['source']))
		return metrics

	def configure_optimizers(self):
		opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
		return {'optimizer': opt,
                'lr_scheduler': torch.optim.lr_scheduler.LinearLR(opt,start_factor=1, end_factor=1/3, total_iters=10000)}

def get_url(model_name):
	'''
	Returns the url of the model to download
	Parameters:
		model_name (str): Name of the model to download

	Returns:
		str: URL of the model
	'''
	if model_name == 'mbart':
		return 'facebook/mbart-large-50-many-to-many-mmt'
	elif model_name == 'm2m':
		return "facebook/m2m100_418M"
	elif model_name == 'flant5':
		return "google/flan-t5-base"
	elif model_name == 'nllb':
		return "facebook/nllb-200-distilled-600M"
	elif model_name == 'llama':
		return "meta-llama/Llama-3.2-1B-Instruct"
	elif model_name == 'qwen':
		return "Qwen/Qwen2.5-VL-7B-Instruct"
	elif model_name == 'eurollm':
		return "utter-project/EuroLLM-1.7B"
	elif model_name == 'gemma':
		return "google/gemma-3-1b-it"
	else:
		print('Model not implemented: {0}'.format(model_name))
		sys.exit(1)

def load_datasets(args):
	'''
	Loads the training and development datasets
	'''
	prompter = get_prompter(args.model_name, args.source, args.target)
	
	shards = [	f"{args.folder}train.{args.source}", 
				f"{args.folder}train.{args.target}"
				]
	training = MosesCorpus(shards[0],shards[1], prompter = prompter)

	shards = [	f"{args.folder}dev.{args.source}",
				f"{args.folder}dev.{args.target}"
				]	
	development = MosesCorpus(shards[0],shards[1], prompter = prompter)
	return training, development

def check_parameters(args):
	args.source_code = check_language_code(args.source) if args.model_name == 'mbart' else args.source
	args.target_code = check_language_code(args.target) if args.model_name == 'mbart' else args.target
	return args

def read_parameters():
	parser = argparse.ArgumentParser(description='Train a model for translation.')
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument('-model','--model_name',default='mbart',choices=NAMES,help='Model to train')
	parser.add_argument('-lora','--lora',action='store_true',help='Whether to use Low-Rank Adaptation or not')
	parser.add_argument("-e","--epochs",type=int,default=3,help="Number of epochs")
	parser.add_argument('-bs','--batch_size',type=int,default=32,help='Batch size')
	parser.add_argument('-lr','--learning_rate',type=float,default=2e-5,help='Learning rate of the optimizer')
	parser.add_argument('-log', '--log_dir', type=str, default='logs', help='Directory to save the logs')

	args = parser.parse_args()
	return args

def main():
	global TOKENIZER, METRIC, MODEL

	args = read_parameters()
	args = check_parameters(args)
	print(args)

	METRIC = load("sacrebleu")
	MODEL, TOKENIZER = load_model(get_url(args.model_name), args, 'cuda')

	if args.lora:
		MODEL = apply_lora(MODEL)

	num_workers = int(os.cpu_count() * 0.75)
	train_dataset, dev_dataset = load_datasets(args)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

	fp16 = not 't5' in args.model_name
	
	translator = TranslationModel(MODEL, TOKENIZER, lr=args.learning_rate)
	callbacks = [L.pytorch.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2, min_delta=0.1),
                L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3, save_weights_only=True,
								  dirpath=f'models/{args.model_name}_{args.source+args.target}')]
	logger = L.pytorch.loggers.TensorBoardLogger(save_dir=args.log_dir, name=f'{args.model_name}_{args.source+args.target}')
	accumulate = 32 // args.batch_size if args.batch_size < 32 else 1
	trainer = L.Trainer(max_epochs=args.epochs,
					 #precision=16 if fp16 else 32,
					 #val_check_interval=0.2,
					 logger=logger,
					 val_check_interval=5000,
					 accumulate_grad_batches=accumulate,
					 default_root_dir=f'models/{args.model_name}_{args.source+args.target}',
					 callbacks=callbacks)
	
	trainer.validate(model=translator,dataloaders=dev_dataloader)
	trainer.fit(model=translator, train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader)
	#trainer.validate(model=translator,dataloaders=dev_dataloader)



if __name__ == '__main__':
	main()
