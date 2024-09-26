"""
Segment-Based Approach with Mbart

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
import argparse
from time import time
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
import restriction as R

device = "cuda:0" if torch.cuda.is_available() else "cpu"
wordTokenizer = TreebankWordTokenizer()

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
	Performs the simulation of the interactive sesion and obtains the WSR and MAR metrics.
	'''
	#try:
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.source)
	if args.final > -1:
		src_lines = read_file(file_name)[:args.final]
	else:
		src_lines = read_file(file_name)
	file_name = '{0}/{1}.{2}'.format(args.folder, args.partition, args.target)
	if args.final > -1:
		trg_lines = read_file(file_name)[:args.final]
	else:
		trg_lines = read_file(file_name)

	if 't5' in args.model_name or args.model_name == 'bloom':
		extend = {'en':'English','fr':'French','de':'German','es':'Spanish', 'gl':'Galician','bn':'Bengali','sw':'Swahili','ne':'Nepali'}
		prompt = f'Translate the following sentence from {extend[args.source]} to {extend[args.target]}: '
		src_lines = [prompt + l for l in src_lines]

	#| PREPARE DOCUMENT TO WRITE
	if args.output:
		file_name = '{0}/{1}.{2}'.format(args.folder,args.output, args.target)
	else:
		file_name = '{0}/sb_imt_{1}.{2}'.format(args.folder, args.model_name, args.target)
	file_out = open(file_name, 'w')
	file_out.write(str(args))
	file_out.write("\n")
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model, tokenizer = R.load_model(model_path, args, device)
	#|=========================================================
	#| PREPARE THE RESTRICTOR
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
		#if i<1280-1:
		#	continue
		# Save the SRC and TRG sentences
		c_src = src_lines[i]
		c_trg = ' '.join(R.tokenize(trg_lines[i],wordTokenizer=wordTokenizer))

		mouse_actions = 0
		word_strokes = 0
		n_words = len(R.tokenize(trg_lines[i],wordTokenizer=wordTokenizer))
		n_chars = len(trg_lines[i])

		# Convert them to ids
		encoded_src = tokenizer(c_src, return_tensors="pt").to(device)
		encoded_trg = [2] + tokenizer(text_target=c_trg).input_ids[:-1]
		if len(encoded_trg) > 512:
			continue

		# Prints
		if args.verbose:
			print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,c_src,c_trg))

		ite = 0
		MAX_TOKENS = 400
		restrictor = Restrictor(VOCAB,tokenizer,len(R.tokenize(c_trg,wordTokenizer=wordTokenizer)))
		ended = False
		ini = time()
		generated_tokens = model.generate(**encoded_src,
								#forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
								max_new_tokens=MAX_TOKENS).tolist()[0]
		output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
		tiempo_total += time() - ini 
		iteraciones += 1
		if len(generated_tokens) >= MAX_TOKENS:
			MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))

		if args.verbose:
			print("ITE {0}: {1}".format(ite, output))
		while not ended:
			# Generate the translation
			ite += 1

			actions, corrections, ended = restrictor.check_segments(c_trg, output, verbose=args.verbose)
			word_strokes += corrections
			mouse_actions += actions
			if args.verbose:
				print('Mouse actions:',actions)
				print('Word strokes:',corrections)

			if not ended:
				restrictor.prepare(args.model_name)

				ini = time()
				raw_output = model.generate(**encoded_src,
								max_new_tokens=MAX_TOKENS,
								prefix_allowed_tokens_fn=restrictor.restrict)
				generated_tokens = raw_output.tolist()[0]
				output = restrictor.decode(generated_tokens)
				tiempo_total += time() - ini
				iteraciones += 1
				if len(generated_tokens) >= MAX_TOKENS:
					MAX_TOKENS = min(512, int(MAX_TOKENS*(3/2)))
				elif len(generated_tokens) > 3/4 * MAX_TOKENS:
					MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))

			if args.verbose:
				print("ITE {0} ({1}): {2}".format(ite, len(generated_tokens), output))
		total_words += n_words
		total_chars += n_chars
		total_ws += word_strokes
		total_ma += mouse_actions

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
	args.source_code = R.check_language_code(args.source) if args.model_name == 'mbart' else args.source

	# Check Target Language
	args.target_code = R.check_language_code(args.target) if args.model_name == 'mbart' else args.target

	# Check the model that is going to load
	if args.model == None:
		args.model = "./mbart-large-50-many-to-many-mmt"

	return args

def read_parameters():
	parser = argparse.ArgumentParser(description='Simulates a user in an IMT task and evaluate the WSR and MAR metrics')
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument("-model", "--model", required=False, help="Model to load")
	parser.add_argument("-out", "--output", required=False, help="Output file")
	parser.add_argument("-seg","--segment_based",action='store_true',help='Whether to use segment-based approach or not. Default to prefix-based.')
	parser.add_argument('-model_name','--model_name', required=False, default='mbart', choices=['mbart','m2m','flant5','nllb','bloom'], help='Model name')
	parser.add_argument('-p','--partition',required=False, default='test', choices=['dev','test'], help='Partition to evaluate')
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
