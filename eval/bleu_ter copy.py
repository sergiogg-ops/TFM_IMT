import argparse

import torch
import evaluate
from restriction import load_model, check_language_code
from transformers import TranslationPipeline, Text2TextGenerationPipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def read_file(name):
	'''
	Opens a file and split the lines into a list

	Parameters:
		name (str): Name of the file to open
	
	Returns:
		list: List with the lines of the file
	'''
	file_r = open(name, 'r')
	lines = file_r.read().splitlines()
	file_r.close()
	return lines

def translate(args):
	'''
	Translate the setences to the target language and evaluate the BLEU and TER metrics
	'''
	src_lines = ['']*100
	trg_lines = ['Hola que tal']*100
	MAX_TOKENS = 400
	bleu_metric = evaluate.load('bleu',trust_remote_code=True)
	ter_metric = evaluate.load('ter',trust_remote_code=True)

	hypothesis = trg_lines[:10]

	print('Evaluando metricas...')
	bleu = [bleu_metric.compute(predictions=[hyp],references=[ref])['bleu'] for hyp, ref in zip(hypothesis, trg_lines[:10])]
	ter = [ter_metric.compute(predictions=[hyp],references=[ref])['score'] for hyp, ref in zip(hypothesis, trg_lines[:10])]
	print('BLEU:')
	print(f'\t{sum(bleu)/len(bleu)}')
	print('TER:')
	print(f'\t{sum(ter)/len(ter)}')

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
	parser = argparse.ArgumentParser(description='Translate and evaluate the BLEU and TER metrics')
	parser.add_argument("-p","--partition", required=False, default="test", choices=["dev","test"], help="Partition to load")
	parser.add_argument("-model", "--model", required=False, help="Model to load")
	parser.add_argument("-model_name", "--model_name", required=False, choices=['mbart','m2m','flant5','nllb','llama','qwen'], help="Model to load")
	parser.add_argument('-b','--batch_size',required=False,default=64,type=int,help='Batch size for the inference')

	args = parser.parse_args()
	return args

def main():
	# Read Parameters
	args = read_parameters()
	print(args)
	
	translate(args)

if __name__ == "__main__":
	main()
