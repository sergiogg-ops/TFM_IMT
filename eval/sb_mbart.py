"""
Segment-Based Approach with Mbart

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
import argparse
import sys

import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
from transformers import (MBart50TokenizerFast, MBartForConditionalGeneration,
                          PhrasalConstraint)
from transformers.generation import Constraint

device = "cuda:0" if torch.cuda.is_available() else "cpu"
wordTokenizer = TreebankWordTokenizer()

class NegativeConstraint(Constraint):
	r"""Abstract base class for all constraints that can be applied during generation.
	It must define how the constraint can be satisfied.

	All classes that inherit Constraint must follow the requirement that

	```py
	completed = False
	while not completed:
		_, completed = constraint.update(constraint.advance())
	```

	will always terminate (halt).
	"""

	def __init__(self, segment, wrong_word, rest_vocab):
		self.first = segment
		self.second = wrong_word
		self.rest_vocab = rest_vocab
		self.seen_first = False
		self.progress = 0
		self.seqlen = len(segment) + 1
		# test for the above condition
		#self.test()

	def test(self):
		"""
		Tests whether this constraint has been properly defined.
		"""
		counter = 0
		completed = False
		while not completed:
			if counter == 1:
				self.reset()
			advance = self.advance()
			if not self.does_advance(advance):
				raise Exception(
					"Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
				)

			stepped, completed, reset = self.update(advance)
			counter += 1

			if counter > 10000:
				raise Exception("update() does not fulfill the constraint.")

		if self.remaining() != 0:
			raise Exception("Custom Constraint is not defined correctly.")

	def advance(self):
		"""
		When called, returns the token that would take this constraint one step closer to being fulfilled.

		Return:
			token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
		"""
		if self.seen_first:
			return self.rest_vocab
		else:
			return self.first[self.progress]

	def does_advance(self, token_id: int):
		"""
		Reads in a token and returns whether it creates progress.
		"""
		if self.seen_first:
			return bool(token_id != self.second)
		else:
			return self.first[self.progress] == token_id

	def update(self, token_id: int):
		"""
		Reads in a token and returns booleans that indicate the progress made by it. This function will update the
		state of this object unlikes `does_advance(self, token_id: int)`.

		This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
		been generated. This becomes important if token_id != desired token (refer to else statement in
		PhrasalConstraint)

		Args:
			token_id(`int`):
				The id of a newly generated token in the beam search.
		Return:
			stepped(`bool`):
				Whether this constraint has become one step closer to being fulfuilled.
			completed(`bool`):
				Whether this constraint has been completely fulfilled by this token being generated.
			reset (`bool`):
				Whether this constraint has reset its progress by this token being generated.
		"""
		if isinstance(token_id, torch.Tensor):
			print('Ayuda')
		if self.progress < len(self.first) and self.first[self.progress] == token_id:
			self.progress += 1
			if self.progress == len(self.first):
				self.seen_first = True
			return True, False, False
		elif self.seen_first:
			fulfilled = token_id != self.second
			self.progress += fulfilled
			return fulfilled, fulfilled, not fulfilled
		else:
			return False, False, False

	def reset(self):
		"""
		Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
		a constraint is abrupted by an unwanted token.
		"""
		self.seen_first = False
		self.progress = 0

	def remaining(self):
		"""
		Returns the number of remaining steps of `advance()` in order to complete this constraint.
		"""
		return self.seqlen - self.progress

	def copy(self, stateful=False):
		"""
		Creates a new instance of this constraint.

		Args:
			stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

		Return:
			constraint(`Constraint`): The same constraint as the one being called from.
		"""
		nueva = NegativeConstraint(self.first, self.second, self.rest_vocab)
		if stateful:
			nueva.seen_first = self.seen_first
			nueva.progress = self.progress
		return nueva

def read_file(name):
	file_r = open(name, 'r')
	lines = file_r.read().splitlines()
	file_r.close()
	return lines

def tokenize(sentence):
	sentence = sentence.replace('…', '...')
	sentence = sentence.replace('´', '\'')
	sentence = sentence.replace('\'', ' \' ')
	sentence = sentence.replace('.', ' . ')
	sentence = sentence.replace('-', ' - ')
	tokens = wordTokenizer.tokenize(sentence)
	for idx, t in enumerate(tokens):
		t = t.replace('``', '"')
		tokens[idx] = t
	return tokens

def check_prefix(target, hyp):
	prefix = []
	correction = 0

	target = tokenize(target)
	hyp = tokenize(hyp)

	for i in range(len(target)):
		if len(hyp)<=i:
			correction = 1
			prefix.append(target[i])
			break
		elif target[i] == hyp[i]:
			prefix.append(target[i])
		else:
			correction = 1
			prefix.append(target[i])
			if target[i] == hyp[i][:len(target[i])] and len(target)>i+1:
				correction = 2
				prefix.append(target[i+1])
			break
	prefix = ' '.join(prefix)
	prefix += ' '
	return prefix, correction

def check_segments_mar(target,hyp):
	target = tokenize(target)
	hyp = tokenize(hyp)

	segments = []; wrong_words = []; buffer = []
	good_segment = False
	count = 0
	while hyp and count < 10:
		while target and hyp and target[0] == hyp[0]:
			#print('LLenando:',target[0])
			# llenar buffer y seguir inspeccionando
			buffer.append(target[0])
			target = target[1:]
			hyp = hyp[1:]
			good_segment = True
		# ¿venimos de procesar un segmento comun? => vaciar buffer
		if good_segment:
			segments.append(buffer)
			buffer = []
			good_segment = False # ya no :(
			# si no es el ultimo token
			if hyp:
				wrong_words.append(hyp[0])
			#print('Segments:',segments)
		# siguiente token comun en la oracion objetivo
		h = 0
		#print('hyp:',hyp)
		while target and hyp and target[0] != hyp[0]:
			#print('hyp:',hyp)
			#print('target:',target)
			while h < len(hyp) and target[0] != hyp[h]:
				h += 1
			#print('h_idx:',h)
			if h == len(hyp):
				target = target[1:]
				h = 0
			else:
				hyp = hyp[h:]
		count += 1
	return segments, wrong_words

def check_segments(target,hyp):
	target = tokenize(target)
	hyp = tokenize(hyp)

	segments = []; buffer = []
	##############################
	# Prefijo
	##############################
	full_end = False
	segments.append([])
	while target and hyp and target[0] == hyp[0]:
		segments[-1].append(target[0])
		target = target[1:]
		hyp = hyp[1:]
		good_segment = True
		full_end = not target
	correction = [target[0]] if target else []
	if len(target)>1 and hyp and target[0] == hyp[0][:len(target[0])]:
		correction.append(target[1])
		target = target[2:]
	#############################
	# Diccionario de palabra-posicion de hyp
	#############################
	hyp_words = {}
	t = h = 0 # indices de target e hyp
	t_length = len(target)
	h_length = len(hyp)
	for idx, w in enumerate(hyp):
		if w not in hyp_words:
			hyp_words[w] = [idx]
		else:
			hyp_words[w].append(idx)
	##############################
	# Segmentos posteriores
	##############################
	good_segment = False
	#count = 10
	while t < t_length and h < h_length:# and count > 0:
		#count -= 1
		while t < t_length and h < h_length and target[t] == hyp[h]:
			# llenar buffer y seguir inspeccionando
			buffer.append(target[t])
			hyp_words[hyp[h]] = hyp_words.get(hyp[h], [-1])[1:]
			t += 1
			h += 1
			good_segment = True
		# ¿venimos de procesar un segmento comun? => vaciar buffer
		if good_segment:
			segments.append(buffer)
			buffer = []
			good_segment = False # ya no :(
			full_end = t >= t_length
			# si no es el ultimo token
			#if t < t_length and not correction:
			#	correction = target[t]
			#print('Segments:',segments)
		# siguiente token comun en la oracion objetivo
		#print('hyp:',hyp)
		while t < t_length and h < h_length and target[t] != hyp[h]:
			#print('hyp:',hyp)
			#print('target:',target)
			next_h = hyp_words.get(target[t], [])
			#print('h_idx:',h)
			if next_h:
				h = next_h[0]
				hyp_words[target[t]] = next_h[1:]
			else:
				t = t + 1
	return segments, correction, full_end

def create_constraints(segments, correction, full_end, tokenizer, min_len = 1, del_punct = False):
	# prefijo
	prefix = segments[0] + correction
	prefix = [2] + tokenizer(text_target=' '.join(prefix)).input_ids[:-1]
	segments = segments[1:]
	# eliminar signos de puntuacion del principio y final de los segmentos
	if del_punct:
		for i in range(1,len(segments)):
			if segments[i]:
				segments[i] = segments[i] if segments[i][-1] not in ['.',',',';',':','!','?'] else segments[i][:-1]
			if segments[i]:
				segments[i] = segments[i] if segments[i][0] not in ['.',',',';',':','!','?'] else segments[i][1:]
	segments = [seg for seg in segments if len(seg) >=  min_len]
	# segmentos intermedios
	tok_segments = []
	if len(segments) > 1:
		tok_segments += [tokenizer.encode(' '.join(s))[1:-1] for s in segments[:-1]]
	# ultimo segmento
	elif segments:
		tok_segments.append(tokenizer.encode(' '.join(segments[-1]))[1:])
		if not full_end:
			tok_segments[-1] = tok_segments[-1][:-1] # quita eos
		
	constraints = [PhrasalConstraint(s) for s in tok_segments]
	return prefix, constraints

def translate(args):
	#try:
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	file_name = '{0}/test.{1}'.format(args.folder, args.source)
	src_lines = read_file(file_name)
	file_name = '{0}/test.{1}'.format(args.folder, args.target)
	trg_lines = read_file(file_name)

	#| PREPARE DOCUMENT TO WRITE
	file_name = '{0}/imt_mbart.{1}'.format(args.folder, args.target)
	file_out = open(file_name, 'w')
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
	tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang=args.source_code, tgt_lang=args.target_code)
	#|========================================================
	#| PREPARE PREFIX FORCER
	prefix = []
	VOCAB = [*range(len(tokenizer))]
	def restrict_prefix(batch_idx, prefix_beam):
		pos = len(prefix_beam)
		if pos<len(prefix):
			return [prefix[pos]]
		return VOCAB
	
	#|=========================================================
	#| GET IN THE RIGHT PLACE

	total_words = 0
	total_chars = 0
	for line in trg_lines[:args.initial]:
		total_words += len(tokenize(line))
		total_chars += len(line)
	total_ws = total_words * args.word_stroke
	total_ma = total_chars * args.mouse_action
	#|=========================================================
	print(args.initial,'/',len(src_lines))
	for i in range(args.initial, len(src_lines)):
		#if i<1280-1:
		#	continue
		# Save the SRC and TRG sentences
		c_src = src_lines[i]
		c_trg = ' '.join(tokenize(trg_lines[i]))

		mouse_actions = 0
		word_strokes = 0
		n_words = len(tokenize(trg_lines[i]))
		n_chars = len(trg_lines[i])

		# Convert them to ids
		encoded_src = tokenizer(c_src, return_tensors="pt").to(device)
		encoded_trg = [2] + tokenizer(text_target=c_trg).input_ids[:-1]

		# Prints
		#print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,c_src,c_trg))

		ite = 0
		prefix = []
		len_old_prefix = 0
		MAX_TOKENS = 128
		constraints = []
		segments = []
		generated_tokens = model.generate(**encoded_src,
								forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
								max_new_tokens=MAX_TOKENS).tolist()[0]
		while len(segments) != 1:
			# Generate the translation
			if len(generated_tokens) >= MAX_TOKENS:
				MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))
			output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

			#print("ITE {0}: {1}".format(ite, output))
			ite += 1

			#prefix, correction = check_prefix(c_trg, output)
			segments, correction, full_end = check_segments(c_trg, output)
			#print(segments)
			if len(segments) != 1:
				prefix, constraints = create_constraints(segments, correction, full_end, tokenizer, min_len=3)

				#print('generando')
				if constraints:
					generated_tokens = model.generate(**encoded_src,
									forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
									max_new_tokens=MAX_TOKENS,
									constraints = constraints,
									prefix_allowed_tokens_fn=restrict_prefix).tolist()[0]
				else:
					generated_tokens = model.generate(**encoded_src,
									forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
									max_new_tokens=MAX_TOKENS,
									prefix_allowed_tokens_fn=restrict_prefix).tolist()[0]
				#print('listo')

				word_strokes += len(correction)
			
			mouse_actions += (len(segments)-1) * 2 + 1

			#file_out.write("{}\n".format(output[0]))
		print("WSR: {0:.4f} MAR: {1:.4f}".format(word_strokes/n_words, mouse_actions/n_chars))
		#print("Total Mouse Actions: {}".format(mouse_actions))
		#print("Total Word Strokes: {}".format(word_strokes))
		total_words += n_words
		total_chars += n_chars
		total_ws += word_strokes
		total_ma += mouse_actions

		#output_txt = "Line {0} T_WSR: {1:.4f} T_MAR: {2:.4f}".format(i, total_ws/total_words, total_ma/total_chars)
		#output_txt = "Line {0} T_MAR: {2:.4f}".format(i, total_ma/total_chars)
		#print(output_txt)
		#print("\n")
		file_out.write("{2} T_WSR: {0:.4f} T_MAR: {1:.4f}\n".format(total_ws/total_words, total_ma/total_chars, i))
		#file_out.write("{2} T_MAR: {1:.4f}\n".format(total_ma/total_chars, i))
		file_out.flush()
	file_out.close()
	#except:

	#	file_out.write("T_WSR: {0:.4f} T_MAR: {1:.4f}\n".format(total_ws/total_words, total_ma/total_chars))
	#	file_out.write("T_MAR: {1:.4f}\n".format(total_ma/total_chars))
	#	file_out.close()

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
	# Check Source Language
	args.source_code = check_language_code(args.source)

	# Check Target Language
	args.target_code = check_language_code(args.target)

	# Check the model that is going to load
	if args.model == None:
		args.model = "./mbart-large-50-many-to-many-mmt"

	return args

def read_parameters():
	parser = argparse.ArgumentParser()
	parser.add_argument("-src", "--source", required=True, help="Source Language")
	parser.add_argument("-trg", "--target", required=True, help="Target Language")
	parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
	parser.add_argument("-model", "--model", required=False, help="Model to load")
	parser.add_argument("-ini","--initial", required=False, default=0, type=int, help="Initial line")
	parser.add_argument("-wsr","--word_stroke", required=False, default=0, type=float, help="Last word stroke ratio")
	parser.add_argument("-mar","--mouse_action", required=False, default=0, type=float, help="Last mouse action ratio")

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
