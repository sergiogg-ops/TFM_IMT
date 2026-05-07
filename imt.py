"""
Segment-Based Approach with Mbart

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
import argparse
from time import time
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
import model as M
import restriction as R


MAX_TOKENS = 512 # Maximum number of tokens to generate
device = "cuda:0" if torch.cuda.is_available() else "cpu"
wordTokenizer = TreebankWordTokenizer()
extend = {'en':'English','fr':'French','de':'German','es':'Spanish', 'gl':'Galician','bn':'Bengali','sw':'Swahili','ne':'Nepali'}

def imt_simulation(model, model_name, restrictor, encoded_src, c_trg, prompter, verbose):
	'''
	Performs the simulation of the IMT task for one sentence

	Parameters:
		model (transformers.model): Model to use
		model_name (str): Codename of the model
		restrictor (restriction.Restriction): Object to restrict the output
		encoded_src (dict): Encoded source sentence
		c_trg (str): Target sentence
		verbose (bool): Whether to show the output or not
	'''
	ite, tiempo_total, iteraciones, word_strokes, mouse_actions = 0, 0, 0, 0, 0
	ended = False
	MAX_TOKENS = 512
	while not ended:
		# Generate the translation
		remove_sos = model_name in M.PROMPTERS
		restrictor.prepare(remove_sos=remove_sos,remove_eos=not remove_sos)
		# printable = [restrictor.tokenizer.convert_ids_to_tokens(t) for t in restrictor.tok_segments]

		ini = time()
		generated_tokens = model.generate(**encoded_src,
						max_new_tokens=MAX_TOKENS,
						prefix_allowed_tokens_fn=restrictor.restrict).tolist()[0]
		output = restrictor.decode(generated_tokens, prompter.decode)
		if verbose:
			print("ITE {0} ({1}): {2}".format(ite, len(generated_tokens), output))
		#if args.model_name in M.PROMPTERS:
		# 	output = output[len(query):]
		tiempo_total += time() - ini
		iteraciones += 1
		if len(generated_tokens) >= MAX_TOKENS:
			MAX_TOKENS = min(512, int(MAX_TOKENS*(3/2)))
		elif len(generated_tokens) > 3/4 * MAX_TOKENS:
			MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))

		actions, corrections, ended = restrictor.check_segments(c_trg, output, verbose=verbose)
		word_strokes += corrections
		mouse_actions += actions
		if verbose:
			print('Mouse actions:',actions)
			print('Word strokes:',corrections)
		ite += 1
	return word_strokes, mouse_actions, tiempo_total, iteraciones
	
def translate(args):
	#try:
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	src_lines, trg_lines = M.load_data(args.folder, args.source, args.target, args.partition)
	if args.model_name in M.PROMPTERS:
		prompter = M.get_prompter(args.model_name, args.source, args.target)
	else:
		prompter = M.Prompter()
	if args.final > -1:
		src_lines = src_lines[:args.final]
		trg_lines = trg_lines[:args.final]

	#| PREPARE DOCUMENT TO WRITE
	if args.output:
		file_name = '{0}/{1}.{2}'.format(args.folder,args.output, args.target)
	else:
		name = 'sb' if args.segment_based else 'pb'
		file_name = f'{args.folder}/{name}_imt_{args.model_name}.{args.target}'
	file_out = open(file_name, 'w')
	file_out.write(str(args))
	file_out.write("\n")
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model, tokenizer = M.load_model(model_path, args, device)
	VOCAB = [*range(len(tokenizer))]
	tiempo_total = 0
	iteraciones = 0
	
	#|=========================================================
	#| GET IN THE RIGHT PLACE
	total_words = 0
	total_chars = 0
	for line in trg_lines[:args.initial]:
		total_words += len(R.tokenize(line,wordTokenizer=wordTokenizer))
		total_chars += len(line)
	total_ws = total_words * args.word_stroke
	total_ma = total_chars * args.mouse_action
	#|=========================================================s	
	Restrictor = R.SegmentRestrictor if args.segment_based else R.PrefixRestrictor
	for i in range(args.initial, len(src_lines)):
		# Save the SRC and TRG sentences
		c_src = src_lines[i]
		c_trg = ' '.join(R.tokenize(trg_lines[i],wordTokenizer=wordTokenizer))

		mouse_actions = 0
		word_strokes = 0
		#n_words = len(R.tokenize(trg_lines[i],wordTokenizer=wordTokenizer))
		n_words = len(R.tokenize(c_trg,wordTokenizer=wordTokenizer))
		n_chars = len(trg_lines[i])

		# Convert them to ids
		query = prompter.src_format(c_src, args.source, args.target)
		encoded_src = tokenizer(query, return_tensors="pt").to(device)
		# encoded_trg = [2] + tokenizer(text_target=c_trg).input_ids[:-1]
		# if len(encoded_trg) > 512:
		# 	continue

		if args.verbose:
			print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,query,c_trg))

		iteraciones += 1
		start_char = 'Ġ' if args.model_name == 'llama' else '▁'
		restrictor = Restrictor(vocab=VOCAB,
						  tokenizer=tokenizer,
						  prompt=query,
						  start=start_char,
						  target_len=n_words)
		word_strokes, mouse_actions, tiempo, iters = imt_simulation(model, 
															  args.model_name, 
															  restrictor, 
															  encoded_src, 
															  c_trg, 
															  prompter,
															  args.verbose)
		total_words += n_words
		total_chars += n_chars
		total_ws += word_strokes
		total_ma += mouse_actions
		tiempo_total += tiempo
		iteraciones += iters

		output_txt = "Line {0} T_WSR: {1:.4f} T_MAR: {2:.4f} TIME: {3:4f}".format(i, total_ws/total_words, total_ma/total_chars, tiempo_total)
		if args.verbose:
			print(output_txt)
			print("\n")
		file_out.write(f'{word_strokes/n_words}\t{mouse_actions/n_chars}\n')
		file_out.flush()
	output_txt = f"TOTAL => WSR: {total_ws/total_words} - MAR: {total_ma/total_chars} - TIME: {tiempo_total/iteraciones}\n"
	file_out.write(output_txt)
	file_out.close()
	print(output_txt)

def check_parameters(args):
	# Check Source Language
	args.source_code = M.check_language_code(args.source) if args.model_name == 'mbart' else args.source

	# Check Target Language
	args.target_code = M.check_language_code(args.target) if args.model_name == 'mbart' else args.target

	return args

def read_parameters():
	parser = argparse.ArgumentParser(description='Simulates a user in an IMT task and evaluates the WSR and MAR metrics')
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument("-model", "--model", required=True, help="Model to load")
	parser.add_argument("-out", "--output", required=False, help="Output file")
	parser.add_argument("-seg","--segment_based",action='store_true', default=False,help='Whether to use segment-based approach or not. Default to prefix-based.')
	parser.add_argument('-model_name','--model_name', required=True, default='mbart', choices=M.NAMES, help='Model name')
	parser.add_argument('-p','--partition',required=False, default='test', choices=['dev','test'], help='Partition to evaluate, default to test')
	parser.add_argument("-ini","--initial", required=False, default=0, type=int, help="Initial line")
	parser.add_argument("-fin","--final",required=False, default=-1,type=int,help="Final Line")
	parser.add_argument("-wsr","--word_stroke", required=False, default=0, type=float, help="Last word stroke ratio")
	parser.add_argument("-mar","--mouse_action", required=False, default=0, type=float, help="Last mouse action ratio")
	parser.add_argument("-v","--verbose", required=False, default=False, action='store_true', help="Verbose mode")

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
