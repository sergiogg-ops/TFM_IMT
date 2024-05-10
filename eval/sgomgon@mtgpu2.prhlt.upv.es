"""
Segment-Based Approach with Mbart

Example of use:
	> python3 imt_bart.py -src es -trg en -dir es-en
"""
import argparse
import sys
from math import ceil

import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          M2M100ForConditionalGeneration, M2M100Tokenizer,
                          MBart50TokenizerFast, MBartForConditionalGeneration,
                          PhrasalConstraint)
from transformers.generation import Constraint

device = "cuda:0" if torch.cuda.is_available() else "cpu"
wordTokenizer = TreebankWordTokenizer()

class Restrictor():
	def __init__(self,vocab, tokenizer, target, filters = [], values = []):
		self.vocab = vocab
		self.tokenizer = tokenizer
		self.prefix = []
		self.correction = ''
		self.segments = []
		self.prev_tseg = -np.ones(len(target)+1,dtype=int)
		self.filters = filters
		self.values = values

	def create_constraints(self, filters=['min_len'], values=[1], verbose=False):

		if verbose:
			if self.prefix:
				print('Segmentos:',[self.prefix] + self.segments)
			else:
				print('Segmentos:',self.segments)
			print('Correcciones:',self.correction)
		self.segments = [self.correction] + self.segments
		#print('Full End:',full_end)
		# segmentos intermedios
		tok_segments = []
		if self.segments:
			#tok_segments += [self.tokenizer.encode(' '.join(s))[1:-1] for s in self.segments[:-1]]
			tok_segments += [self.tokenizer.encode(s)[1:-1] for s in self.segments] # quita bos y eos
			
		constraints = [PhrasalConstraint(s) for s in tok_segments]
		return constraints
	
	def restrict_prefix(self,batch_idx, prefix_beam):
		pos = len(prefix_beam)
		if pos<len(self.tok_prefix):
			return [self.tok_prefix[pos]]
		return self.vocab

	def check_segments(self,tgt,hyp):
		'''
		Encuentra los segmentos comunes entre dos cadenas de texto

		Parameters:
			tgt (str): Cadena de texto 1 (target)
			hyp (str): Cadena de texto 2 (hypothesis)
		
		Returns:
			mouse_actions (int): numero de nuevos segmentos encontrados
			len(correction) (int): numero de palabras que se han usado para extender el prefijo
			all_right (bool): la hipotesis es correcta o no
			full_end (bool): el ultimo segmento acaba con la oracion objetivo o no
		'''
		tgt = tokenize(tgt)
		hyp = tokenize(hyp)

		lent, lenh = len(tgt), len(hyp)
		#dp = np.zeros((lent+1,lenh+1),dtype=int)
		t_seg = np.zeros(lent+1,dtype=int) # +1 simboliza el final
		# Calcular matriz de cruces
		for i in range(lent):
			j = 0
			while j < lenh and tgt[i] != hyp[j]:
				j += 1
			t_seg[i] = j < lenh
		
		print('Validado:',self.prev_tseg)
		print('Nuevo:   ',t_seg)
		# Calcular acciones a partir de los segmentos nuevos/antiguos
		self.prefix = ''
		i = 0
		while i < lent and t_seg[i] == 1:
			self.prefix += tgt[i] + ' '
			i += 1
		num_corrections = 0
		self.correction = ''
		if i < lent:
			self.correction += tgt[i] + ' '
			i += 1
			num_corrections += 1
		if i < lent and tgt[i-1] == hyp[i-1][:len(tgt[i-1])]:
			self.correction += tgt[i] + ' '
			i += 1
			num_corrections += 1

		self.tok_prefix = [2] + self.tokenizer(text_target=self.prefix).input_ids[:-1]
		#print('Prefijo:',self.prefix)
		t_seg[i:] = self.filter_segments(t_seg[i:])

		self.segments = ['']
		mouse_actions = min(2,i-1) # prefijo (1 o mas palabras), las correcciones se cuentan al final
		first_tok = i-1
		#print(i,'/',lent)
		for j in range(i,lent+1):
			#if t_seg[j] - self.prev_tseg[j] != t_seg[j-1] - self.prev_tseg[j-1]:
			#	mouse_actions += 2 if j - first_tok > 1 else 1
			if t_seg[j-1] == 0 and t_seg[j] == 1:
				first_tok = j
				self.segments.append(tgt[j])
			elif t_seg[j] == 1:
				self.segments[-1] += ' ' + tgt[j]
			elif t_seg[j-1] == 1 and t_seg[j] == 0 and np.sum(t_seg[first_tok:j]) != np.sum(self.prev_tseg[first_tok:j]):
				#print(first_tok,j)
				mouse_actions += 2 if j - first_tok > 1 else 1
		if self.segments[0] == '':
			self.segments = self.segments[1:]

		self.prev_tseg = t_seg
		print('Actions:',mouse_actions)
		# utilizo la lista segments para asegurarme de que el ultimo segmento introducido es
		# el que potencialmente acaba con target (tgt)
		return mouse_actions, num_corrections, t_seg[-1], i == lent
	
	def check_segments2(self,tgt,hyp,verbose=False):
		'''
		Encuentra los segmentos comunes entre dos cadenas de texto

		Parameters:
			tgt (str): Cadena de texto 1 (target)
			hyp (str): Cadena de texto 2 (hypothesis)
		
		Returns:
			mouse_actions (int): numero de nuevos segmentos encontrados
			len(correction) (int): numero de palabras que se han usado para extender el prefijo
			full_end (bool): el ultimo segmento acaba con la oracion objetivo o no
			all_right (bool): la hipotesis es correcta o no
		'''
		tgt = tokenize(tgt)
		hyp = tokenize(hyp)

		lent, lenh = len(tgt), len(hyp)
		#dp = np.zeros((lent+1,lenh+1),dtype=int)
		t_seg = -np.ones(lent+1,dtype=int) # +1 simboliza el final
		#h_seg = np.zeros(lenh+1,dtype=int) # +1 simboliza el final
		dp = np.zeros((lent+1,lenh+1),dtype=int)
		h_max = -np.ones(lenh+1,dtype=int)
		# Calcular matriz de cruces
		for i in range(0,lent):
			for j in range(0,lenh):
				if tgt[i] == hyp[j]:
					dp[i+1,j+1] = dp[i,j] + 1
			#idx = np.argmax(dp[i+1,:]) 
			#t_seg[(i-length+1):i+1] = np.arange(idx-length,idx) if dp[i,idx] > 0 else -1
			prev_t, h = self.search_idx(dp,i+1,h_max)
			#print(prev_t,h)
			if prev_t != -1:
				h_max[h] = i+1 # actualizar usados
				prev_length = dp[prev_t,h]
				t_seg[(prev_t-prev_length+1):prev_t+1] = -1 # eliminar segmento menor anterior
			if h != -1:
				h_max[h] = i+1 # actualizar usados
				length = dp[i+1,h]
				t_seg[(i-length+1):i+1] = np.arange(h-length,h) # incluir segmento mayor actual
			
		if verbose: 
			print('Validado:',self.prev_tseg)
			print('Nuevo:   ',t_seg)

		# Calcular acciones de nuevos segmentos
		self.prefix = ''
		mouse_actions = 0
		i = 0
		while i < lent and t_seg[i] != -1:
			if t_seg[i] != t_seg[i-1]+1:
				mouse_actions += 1 #union de segmentos
			self.prefix += tgt[i] + ' '
			i += 1
		if mouse_actions > 0:
			mouse_actions += 1 # union de segmentos
		mouse_actions += min(2,i) # prefijo (1 o mas palabras)

		prev_correction = self.correction
		self.correction = ''
		num_corrections = 0
		if i < lent:
			self.correction += tgt[i] + ' '
			i += 1
			num_corrections += 1
		if i < lent and tgt[i-1] == hyp[i-1][:len(tgt[i-1])]:
			self.correction += tgt[i] + ' '
			i += 1
			num_corrections += 1
		while i < len(tgt) and len(prev_correction) >= len(self.correction) and prev_correction[:len(self.correction)] == self.correction:
			self.correction += tgt[i] + ' '
			num_corrections += 1
			i += 1
		
		mouse_actions += num_corrections
		self.tok_prefix = [2] + self.tokenizer(text_target=self.prefix).input_ids[:-1]
		t_seg[i:] = self.filter_segments(t_seg[i:])

		first_tok = i-1
		self.segments = ['']
		for j in range(i,lent+1):
			# token validado
			if t_seg[j] != -1:
				# Inicio absoluto de segmento
				if t_seg[j-1] == -1:
					self.segments.append('')
					first_tok = j
				self.segments[-1] += tgt[j] + ' '
			# acciones de raton
			if t_seg[j-1] != -1 and t_seg[j-1] + 1 != t_seg[j]:
				# ¿segmento nuevo?
				if np.sum(t_seg[first_tok:j] != -1) != np.sum(self.prev_tseg[first_tok:j] != -1):
					mouse_actions += 2 if j - first_tok > 1 else 1
				# ¿punto de union de segmentos?
				if t_seg[j] != -1:
					mouse_actions += 2
				first_tok = j
		if self.segments and self.segments[0] == '':
			self.segments = self.segments[1:]
		
		self.prev_tseg = t_seg
		if verbose:
			print('Actions:',mouse_actions)
		return mouse_actions, num_corrections, t_seg[-1], i == lent
	
	def check_segments3(self,tgt,hyp,verbose=False):
		tgt = tokenize(tgt)
		hyp = tokenize(hyp)

		lent, lenh = len(tgt), len(hyp)
		dp = self.cruce(tgt,hyp,lent,lenh)
		print(dp)
		t_seg = self.segmentos(dp, -np.ones(lent+1,dtype=int), 0, lent, 0, lenh)
		if verbose: 
			print('Validado:',self.prev_tseg)
			print('Nuevo:   ',t_seg)

		# PREFIJO
		i = 0
		self.prefix = ''
		while i < lent and t_seg[i] == 1:
			self.prefix += tgt[i] + ' '
			i += 1
		mouse_actions = min(2,i) # prefijo (1 o mas palabras)
		self.tok_prefix = [2] + self.tokenizer(text_target=self.prefix).input_ids[:-1]
		t_seg[i:] = self.filter_segments(t_seg[i:])
		# CORRECCION
		num_corrections = 0
		prev_correction = self.correction
		self.correction = ''
		if i < lent:
			self.correction += tgt[i] + ' '
			i += 1
			num_corrections += 1
		if i < lent and tgt[i-1] == hyp[i-1][:len(tgt[i-1])]:
			self.correction += tgt[i] + ' '
			i += 1
			num_corrections += 1
		while i < lent and len(prev_correction) >= len(self.correction) and prev_correction[:len(self.correction)] == self.correction:
			self.correction += tgt[i] + ' '
			num_corrections += 1
			i += 1
		mouse_actions += 1
		# SEGMENTOS
		self.segments = ['']
		first_tok = i-1
		for j in range(i,lent+1):
			# token validado
			if t_seg[j] != -1:
				# Inicio absoluto de segmento
				if t_seg[j-1] == -1:
					self.segments.append('')
					first_tok = j
				self.segments[-1] += tgt[j] + ' '
			# acciones de raton
			if t_seg[j-1] != -1 and t_seg[j-1] + 1 != t_seg[j]:
				# ¿segmento nuevo?
				if np.sum(t_seg[first_tok:j] != -1) != np.sum(self.prev_tseg[first_tok:j] != -1):
					mouse_actions += 2 if j - first_tok > 1 else 1
				# ¿punto de union de segmentos?
				if t_seg[j] != -1:
					mouse_actions += 2
				first_tok = j
		if self.segments and self.segments[0] == '':
			self.segments = self.segments[1:]
		
		self.prev_tseg = t_seg
		if verbose:
			print('Actions:',mouse_actions)
		return mouse_actions, num_corrections, i == lent

	
	def cruce(self,tgt,hyp,lent,lenh):
		dp = np.zeros((lent+1,lenh+1),dtype=int)
		for i in range(0,lent):
			for j in range(0,lenh):
				if tgt[i] == hyp[j]:
					dp[i+1,j+1] = dp[i,j] + 1
		return dp
	
	def segmentos(self,dp, tgt, ini_t, fin_t, ini_h, fin_h):
		if ini_t > fin_t or ini_h > fin_h:
			return tgt
		fin_lcs = np.unravel_index(np.argmax(dp[ini_t:fin_t,ini_h:fin_h]), (fin_t-ini_t,fin_h-ini_h))
		print(fin_lcs)
		if dp[fin_lcs[0],fin_lcs[1]] == 0:
			return tgt
		ini_lcs = fin_lcs - dp[fin_lcs]
		# segmentos anteriores
		if ini_lcs[0] > ini_t:
			print('anteriores')
			tgt = self.segmentos(dp,tgt,ini_t,ini_lcs[0],ini_h,ini_lcs[1])
		# segmento mas largo comun
		tgt[ini_lcs[0]:fin_lcs[0]] = np.arange((ini_lcs[1],fin_lcs[1]+1))
		# segmentos posteriores
		if fin_lcs[0] < fin_t:
			print('posteriores')
			tgt = self.segmentos(dp,tgt,fin_lcs[0],fin_t,fin_lcs[1],fin_h)
		return tgt
			
	def dicc_idx(self,lista,ini=0):
		dicc = {}
		for i in range(ini,len(lista)):
			if lista[i] in dicc:
				dicc[lista[i]].append(i)
			else:
				dicc[lista[i]] = [i]
		return dicc

	
	def search_idx(self,dp, i,h_max):
		'''
			Buscar indice no conflictivo de mayor longitud

			Parametros:
				dp (np.array): matriz de cruces
				i (int): indice de la oracion objetivo (en target)
				h_max (np.array): lista de indices de las hipotesis (en hypothesis)
			
			Returns:
				indice en target a borrar
				indice en hypothesis
		'''
		posible = np.argsort(dp[i,:]) # hyp idxs
		j = posible.shape[0]-1
		while j >= 0:
			if h_max[posible[j]] == -1:
				return -1, posible[j]
			if dp[h_max[posible[j]],posible[j]] < dp[i,posible[j]]:
				return h_max[posible[j]], posible[j]
			j -= 1
		return -1, -1

	def filter_segments(self, mask):
		for f,v in zip(self.filters,self.values):
			if f == 'min_len':
				mask = self.min_len_filter(mask, v)
			if f == 'max_near':
				mask = self.max_near_filter(mask, v)
			if f == 'max_far':
				mask = self.max_far_filter(mask, v)
		return mask
	
	def min_len_filter(self, mask, min_len):
		ini_seg = 0
		for i in range(1,mask.shape[0]):
			if mask[i] == 0:
				if i - ini_seg < min_len:
					mask[ini_seg:i] = 0
			elif mask[i-1] == 0 and mask[i] == 1:
				ini_seg = i
		return mask
	
	def max_near_filter(self, mask, num):
		i = 1
		while i < mask.shape[0] and num > 0:
			i += 1
			num -= mask[i-1] == 1 and mask[i] == 0
		mask[i:] = 0
		return mask

	def max_far_filter(self, mask, num):
		i = mask.shape[0]-2
		while i >= 0 and num > 0:
			i -= 1
			num -= mask[i] == 1 and mask[i+1] == 0
		mask[:i+1] = 0
		return mask

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

def filter_segments(segments, filters=['min_len'], values=[1], del_punct=False):
	for f,v in zip(filters,values):
		if f == 'min_len':
			segments = [seg for seg in segments if len(seg) >= v]
		if f == 'max_seg':
			if len(segments) > v:
				segments = sorted(segments, key=lambda x: len(x))
				segments = segments[-v:]
		if f == 'max_near':
			segments = segments[:v]
		if f == 'max_far':
			segments = segments[v:]
	if del_punct:
		for i in range(1,len(segments)):
			if segments[i]:
				segments[i] = segments[i] if segments[i][-1] not in ['.',',',';',':','!','?'] else segments[i][:-1]
			if segments[i]:
				segments[i] = segments[i] if segments[i][0] not in ['.',',',';',':','!','?'] else segments[i][1:]
	return segments

def load_model(model_path, args, _dev=None):
	if args.model_name == 'mbart':
		_mdl = MBartForConditionalGeneration.from_pretrained(model_path).to(_dev)
		_tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'm2m':
		_mdl = M2M100ForConditionalGeneration.from_pretrained(model_path).to(_dev)
		_tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'flant5':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
		_tok = AutoTokenizer.from_pretrained("google/flan-t5-small",src_lang=args.source_code, tgt_lang=args.target_code)
	else:
		print('Model not implemented: {0}'.format(args.model_name))
		sys.exit(1)
	return _mdl, _tok
	
def translate(args):
	#try:
	#|========================================================
	#| READ SOURCE AND TARGET DATASET
	file_name = '{0}/test.{1}'.format(args.folder, args.source)
	src_lines = read_file(file_name)
	file_name = '{0}/test.{1}'.format(args.folder, args.target)
	trg_lines = read_file(file_name)

	#| PREPARE DOCUMENT TO WRITE
	if args.output:
		file_name = '{0}/{1}.{2}'.format(args.folder,args.output, args.target)
	else:
		file_name = '{0}/imt_mbart.{1}'.format(args.folder, args.target)
	file_out = open(file_name, 'w')
	file_out.write(str(args))
	file_out.write("\n")
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model_path = args.model
	model, tokenizer = load_model(model_path, args, device)
	#|=========================================================
	#| PREPARE THE RESTRICTOR
	VOCAB = [*range(len(tokenizer))]
	
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
		#encoded_trg = [2] + tokenizer(text_target=c_trg).input_ids[:-1]

		# Prints
		if args.verbose:
			print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,c_src,c_trg))

		ite = 0
		MAX_TOKENS = 128
		restrictor = Restrictor(VOCAB,tokenizer,c_trg,filters=args.filters,values=args.values)
		ended = False
		generated_tokens = model.generate(**encoded_src,
								forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
								max_new_tokens=MAX_TOKENS).tolist()[0]
		if len(generated_tokens) >= MAX_TOKENS:
			MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))
		output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

		if args.verbose:
			print("ITE {0}: {1}".format(ite, output))
		while not ended:
			# Generate the translation
			ite += 1

			#prefix, correction = check_prefix(c_trg, output)
			actions, corrections, ended = restrictor.check_segments3(c_trg, output,verbose=args.verbose)
			#segments, correction, full_end = check_segments(c_trg, output)
			#print(segments)
			word_strokes += corrections
			mouse_actions += actions + 1
			if args.verbose:
				print('Mouse actions:',actions + 1)
				print('Word strokes:',corrections)

			if not ended:
				#prefix, constraints = create_constraints(segments, correction, full_end, tokenizer,filters=['max_near'], values=[3])
				constraints = restrictor.create_constraints(filters=[], values=[],verbose=args.verbose)

				#print('generando')
				if constraints:
					generated_tokens = model.generate(**encoded_src,
									forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
									max_new_tokens=MAX_TOKENS,
									constraints = constraints,
									prefix_allowed_tokens_fn=restrictor.restrict_prefix).tolist()[0]
				else:
					generated_tokens = model.generate(**encoded_src,
									forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
									max_new_tokens=MAX_TOKENS,
									prefix_allowed_tokens_fn=restrictor.restrict_prefix).tolist()[0]
				if len(generated_tokens) >= MAX_TOKENS:
					MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))
				output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

				if args.verbose:
					print("ITE {0}: {1}".format(ite, output))
				#print('listo')

			#file_out.write("{}\n".format(output[0]))
		total_words += n_words
		total_chars += n_chars
		total_ws += word_strokes
		total_ma += mouse_actions
		
		#print("WSR: {0:.4f} MAR: {1:.4f}".format(word_strokes/n_words, mouse_actions/n_chars))
		#print("Total Mouse Actions: {}".format(mouse_actions))
		#print("Total Word Strokes: {}".format(word_strokes))

		output_txt = "Line {0} T_WSR: {1:.4f} T_MAR: {2:.4f}".format(i, total_ws/total_words, total_ma/total_chars)
		#output_txt = "Line {0} T_MAR: {2:.4f}".format(i, total_ma/total_chars)
		if args.verbose:
			print('Word strokes:',corrections)
			print('Mouse actions:',actions + 1)
			print(output_txt)
			print("\n")
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
	parser.add_argument("-out", "--output", required=False, help="Output file")
	parser.add_argument('-model_name','--model_name', required=False, default='mbart', choices=['mbart','m2m','flant5'], help='Model name')
	parser.add_argument("-ini","--initial", required=False, default=0, type=int, help="Initial line")
	parser.add_argument("-wsr","--word_stroke", required=False, default=0, type=float, help="Last word stroke ratio")
	parser.add_argument("-mar","--mouse_action", required=False, default=0, type=float, help="Last mouse action ratio")
	parser.add_argument("-f","--filters", required=False, nargs='+', default=[],choices=['min_len','max_near','max_far'], help="Filters to apply to the segments")
	parser.add_argument("-val","--values", required=False, nargs='+', default=[], type=int, help="Values for the filters")
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
