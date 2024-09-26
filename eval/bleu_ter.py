import argparse

import torch
import evaluate
from restriction import load_model, check_language_code
from transformers import TranslationPipeline

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
	print('Cargando modelo...')
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.source)
	src_lines = read_file(file_name)
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.target)
	trg_lines = read_file(file_name)
	if 't5' in args.model_name:
		extend = {'en':'English','fr':'French','de':'German','es':'Spanish', 'gl':'Galician','bn':'Bengali','sw':'Swahili','ne':'Nepali'}
		prompt = f'Translate the following sentence from {extend[args.source]} to {extend[args.target]}: '
		src_lines = [prompt + l for l in src_lines]
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model, tokenizer = load_model(model_path, args, device)
	#|========================================================
	MAX_TOKENS = 400
	bleu_metric = evaluate.load('bleu',trust_remote_code=True)
	ter_metric = evaluate.load('ter',trust_remote_code=True)
	translator = TranslationPipeline(model=model,tokenizer=tokenizer, batch_size=args.batch_size, device=device)
	#|========================================================
	#| TRANSLATE
	print('Traduciendo...')
	hypothesis = translator(src_lines, src_lang=args.source_code, tgt_lang=args.target_code, max_length=MAX_TOKENS)
	hypothesis = [t['translation_text'] for t in hypothesis]

	print('Evaluando metricas...')
	bleu = [bleu_metric.compute(predictions=[hyp],references=[ref])['bleu'] for hyp, ref in zip(hypothesis, trg_lines) if len(hyp.strip()) > 0 and len(ref.strip()) > 0]
	ter = [ter_metric.compute(predictions=[hyp],references=[ref])['score'] for hyp, ref in zip(hypothesis, trg_lines) if len(hyp.strip()) > 0 and len(ref.strip()) > 0]
	print('BLEU:')
	print(f'\t{sum(bleu)/len(bleu)}')
	print('TER:')
	print(f'\t{sum(ter)/len(ter)}')
	with open(f'{args.folder}/{args.model_name}.{args.target}', 'w') as file:
		for b, t in zip(bleu,ter):
			file.write(f'{b}\t{t}\n')

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
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where the dataset is")
	parser.add_argument("-p","--partition", required=False, default="test", choices=["dev","test"], help="Partition to load")
	parser.add_argument("-model", "--model", required=False, help="Model to load")
	parser.add_argument("-model_name", "--model_name", required=False, choices=['mbart','m2m','flant5','nllb'], help="Model to load")
	parser.add_argument('-b','--batch_size',required=False,default=64,type=int,help='Batch size for the inference')

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
