import argparse

import torch
import evaluate
import model as M
from transformers import TranslationPipeline, Text2TextGenerationPipeline
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def translate(args):
	'''
	Translate the setences to the target language and evaluate the BLEU and TER metrics
	'''
	print('Cargando modelo...')
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	src_lines, trg_lines = M.load_data(args.folder, args.source, args.target, args.partition)
	prompter = M.get_prompter(args.model_name, args.source, args.target)
	src_lines = [prompter.src_format(l) for l in src_lines]
	# src_lines = src_lines[:10]
	# trg_lines = trg_lines[:10]
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model, tokenizer = M.load_model(model_path, args, device)
	#|========================================================
	MAX_TOKENS = 400
	bleu_metric = evaluate.load('bleu',trust_remote_code=True)
	ter_metric = evaluate.load('ter',trust_remote_code=True)
	print('Traduciendo...')
	if args.model_name in M.PROMPTERS:
		outputs = []
		# Process inputs in batches
		for i in tqdm(range(0, len(src_lines), args.batch_size)):
			model.to(device)
			batch = src_lines[i:i + args.batch_size]
			input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
			with torch.no_grad():
				output = model.generate(**input_ids, max_new_tokens=MAX_TOKENS, pad_token_id=tokenizer.pad_token_id)
			decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=False)
			outputs.extend(decoded_outputs)
		with open('output.txt','w') as f:
			f.write('\n-------------------------------------\n'.join(outputs))
		hypothesis = [prompter.clean(o) for o in outputs]
		with open('hyp.txt','w') as f:
			f.write('\n'.join(hypothesis))
	else:
		translator = TranslationPipeline(model=model,tokenizer=tokenizer, batch_size=args.batch_size, device=device)
		hypothesis = translator(src_lines, src_lang=args.source_code, tgt_lang=args.target_code, max_length=MAX_TOKENS)
		hypothesis = [t['translation_text'] for t in hypothesis]

	#print(hypothesis)
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
	args.source_code = M.check_language_code(args.source) if args.model_name == 'mbart' else args.source

	# Check Target Language
	args.target_code = M.check_language_code(args.target) if args.model_name == 'mbart' else args.target

	return args

def read_parameters():
	parser = argparse.ArgumentParser(description='Translate and evaluate the BLEU and TER metrics. The metrics for each sentence are saved in a file called as the model name')
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where the dataset is")
	parser.add_argument("-p","--partition", required=False, default="test", choices=["dev","test"], help="Partition to load")
	parser.add_argument("-model", "--model", required=True, help="Model to load")
	parser.add_argument("-model_name", "--model_name", required=True, choices=M.NAMES, help="Model to load")
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
