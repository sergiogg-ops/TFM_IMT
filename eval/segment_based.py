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
#np.set_printoptions(threshold=np.inf)

class OrderedListConstraints(Constraint):
	def __init__(self, segments):
		self.tok_segments = segments
		self.constraints = [PhrasalConstraint(seg) for seg in segments]
		self.num_seg = len(segments)
		self.idx_seg = 0
		self.seqlen = sum([c.seqlen for c in self.constraints])
		self.completed = False
		self.progress = 0

	def advance(self):
		"""
    	When called, returns the token that would take this constraint one step closer to being fulfilled.

		Return:
			token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
		"""
		if self.completed:
			return None
		return self.constraints[self.idx_seg].advance()

	def does_advance(self, token_id: int):
		"""
		Reads in a token and returns whether it creates progress.
		"""
		if self.completed:
			return False
		return self.constraints[self.idx_seg].does_advance(token_id)

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
		stepped, completed, reset = self.constraints[self.idx_seg].update(token_id)
		if stepped:
			self.progress += 1
		if completed:
			self.idx_seg += 1
			self.completed = self.idx_seg >= self.num_seg
		if self.idx_seg == 0:
			return stepped,self.completed,reset
		else:
			return stepped, self.completed, False

	def reset(self):
		"""
		Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
		a constraint is abrupted by an unwanted token.
		"""
		for c in self.constraints[:self.idx_seg]:
			c.reset()
		self.idx_seg = 0
		self.completed = False
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
		return OrderedListConstraints(self.tok_segments)

class Restrictor():
	def __init__(self,vocab, tokenizer, target_len, filters = [], values = [], wait_tokens = 3):
		self.mierdaenbote=0
		# Vocabulario
		self.vocab = vocab
		self.tokenizer = tokenizer
		self.start_toks = [value for key,value in self.tokenizer.get_vocab().items() if key[0] == '▁']
		#self.start_toks = [value for key,value in self.tokenizer.get_vocab().items() if not key[0].isalpha()]
		self.eos = self.tokenizer.get_vocab()[tokenizer.eos_token]
		if self.eos in self.start_toks:
			self.start_toks.remove(self.eos)
		# segmentos
		self.prefix = []
		self.correction = ''
		self.segments = []
		self.prev_tseg = -np.ones(target_len+1,dtype=int)
		self.prev_ini = np.zeros(target_len+1,dtype=int)
		# filtrado de segmentos
		self.filters = filters
		self.values = values
		# forzado de busqueda
		self.max_wait = wait_tokens

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
	def create_constraints2(self, filters=['min_len'], values=[1], verbose=False):
		if verbose:
			if self.prefix:
				print('Segmentos:',[self.prefix] + self.segments)
			else:
				print('Segmentos:',self.segments)
			print('Correcciones:',self.correction)
		self.segments = [self.correction] + self.segments
		print(self.segments)
		tok_segments = [self.tokenizer.encode(s)[1:-1] for s in self.segments] # quita bos y eos
		print([self.tokenizer.decode(s) for s in tok_segments])
			
		return [OrderedListConstraints(tok_segments)]

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
		# tokenizar strings
		tgt = tokenize(tgt)
		hyp = tokenize(hyp)

		lent, lenh = len(tgt), len(hyp)
		dp = self.cruce(tgt,hyp,lent,lenh)
		#print(dp)
		t_seg = self.get_segments(dp)
		ini_seg = np.zeros(t_seg.shape[0],dtype=int)
		if verbose: 
			print('Validado:',self.prev_tseg)
			print('Nuevo:   ',t_seg)
		
		self.segments = ['']
		num_corrections = 0
		mouse_actions = 0

		# SEGMENTOS
		first_tok = -1
		new_seg = not self.prev_ini[0]
		for j in range(lent+1):
			# token validado
			if t_seg[j] != -1:
				# Inicio absoluto de segmento o inicio de segmento ya validado consecutivo
				if t_seg[j-1] == -1:
					self.segments.append('')
					first_tok = j
					ini_seg[j] = 1
				elif self.prev_ini[j] and t_seg[j-1] + 1 == t_seg[j]:
					if new_seg:
						new_seg = False
					else:
						self.segments.append('')
						first_tok = j
						ini_seg[j] = 1
				self.segments[-1] += tgt[j] + ' '
			else:
				new_seg = True
			# acciones de raton
			if (t_seg[j-1] != -1 and t_seg[j-1] + 1 != t_seg[j]) or (self.prev_tseg[j-1] == -1 and t_seg[j-1] != -1 and ini_seg[j]):
				# ¿segmento nuevo?
				if np.sum(t_seg[first_tok:j] != -1) != np.sum(self.prev_tseg[first_tok:j] != -1):
					mouse_actions += 2 if j - first_tok > 1 else 1
				# ¿punto de union de segmentos?
				if t_seg[j] != -1:
					mouse_actions += 2
				first_tok = j
		while self.segments and self.segments[0] == '':
			self.segments = self.segments[1:]
		# CORRECCION
		i = 0
		pos = 0
		while t_seg[i] != -1:
			pos += ini_seg[i]
			i += 1
		self.correction = ''
		if i < lent:
			self.correction += tgt[i] + ' '
			t_seg[i] = lenh
			ini_seg[i] = 1
			i += 1
			num_corrections += 1
			mouse_actions += 1
			self.segments = self.segments[:pos] + [self.correction] + self.segments[pos:]

		self.prev_tseg = t_seg
		self.prev_ini = ini_seg
		if verbose:
			print('inicios:',self.prev_ini)
			print('Actions:',mouse_actions)
			print('Segmentos:', self.segments)
			print('Correccion:', self.correction)
		return mouse_actions, num_corrections, i-num_corrections == lent

	
	def cruce(self,tgt,hyp,lent,lenh):
		dp = np.zeros((lent+1,lenh+1),dtype=int)
		for i in range(0,lent):
			for j in range(0,lenh):
				if tgt[i] == hyp[j]:
					dp[i+1,j+1] = dp[i,j] + 1
		return dp
	
	def get_segments(self,dp):
		#print('prev',self.prev_tseg)
		t_seg = -np.ones(dp.shape[0],dtype=int)
		length = int(self.prev_tseg[0] != -1)
		last_seg = 0
		for i in range(1,dp.shape[0]):
			# final de segmento
			if self.prev_tseg[i-1] != -1 and (self.prev_tseg[i-1] + 1 != self.prev_tseg[i] or self.prev_ini[i]):
				pos = np.argmax(dp[i])
				#print(i,pos,length)
				t_seg = self.lcs(dp,t_seg,last_seg,i-length+1,pos-length,pos)
				t_seg[i-length:i] = np.arange(pos-length,pos)
				last_seg = i
				length = 0
			length += self.prev_tseg[i] != -1
		t_seg = self.lcs(dp, t_seg, last_seg, dp.shape[0],max(0,t_seg[last_seg]), dp.shape[1])
		return t_seg
	
	def lcs(self,dp, tgt, ini_t, fin_t, ini_h, fin_h):
		#print('fragmento',(ini_t, fin_t), (ini_h, fin_h))
		if ini_t >= fin_t or ini_h >= fin_h:
			return tgt
		fin_lcs = np.unravel_index(np.argmax(dp[ini_t:fin_t,ini_h:fin_h]), (fin_t-ini_t,fin_h-ini_h))
		fin_lcs = (fin_lcs[0]+ini_t,fin_lcs[1]+ini_h)
		if dp[fin_lcs[0],fin_lcs[1]] == 0:
			return tgt
		ini_lcs = fin_lcs - dp[fin_lcs]
		# segmento mas largo comun
		tgt[ini_lcs[0]:fin_lcs[0]] = np.arange(ini_lcs[1],fin_lcs[1])
		# segmentos anteriores
		if ini_lcs[0] >= ini_t:
			tgt = self.lcs(dp,tgt,ini_t,ini_lcs[0]+1,ini_h,ini_lcs[1]+1)
		# segmentos posteriores
		if fin_lcs[0] < fin_t:
			tgt = self.lcs(dp,tgt,fin_lcs[0]+1,fin_t,fin_lcs[1]+1,fin_h)
		return tgt

	
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
	
	def prepare(self):
		self.segments = [self.tokenizer.encode(s)[1:-1] for s in self.segments]
	
	def restrict_prefix(self,batch_idx, prefix_beam):
		pos = len(prefix_beam)
		if pos<len(self.tok_prefix):
			return [self.tok_prefix[pos]]
		return self.vocab
	
	def restrict(self,batch_id, input_ids):
		#if self.mierdaenbote % 10 == 0:
		#	print(self.segments)
		self.mierdaenbote += 1
		idx_seg, idx_tok,last_match = self.get_state(input_ids)
		#print(idx_seg, idx_tok,last_match,len(input_ids))
		#print(input_ids.tolist())
		waiting = len(input_ids) - last_match
		# ¿Se ha terminado de añadir segmentos?
		if idx_seg >= len(self.segments):
			#print('con eos###################################################################')
			return self.vocab	
		
		# ¿hemos terminado de añadir el segmento actual?
		# ultimo token de segmento -> tokens de inicio de palabra
		if idx_tok >= len(self.segments[idx_seg]):
			if idx_seg >= len(self.segments)-1:
				# ultimo token añadido
				#print('con eos###################################################################')
				return self.start_toks + [self.eos]
			#print('start tokens')
			return self.start_toks
		
		token = None
		# Si no estamos añadiendo ningun segmento miramos si podemos empezar con uno
		if idx_tok == -1:
			if waiting >= self.max_wait or input_ids[-1] == self.segments[idx_seg][0]:
				# ¿el token sugerido es el inicio del siguiente segmento?
				token = self.segments[idx_seg][0]
		else:
			token = self.segments[idx_seg][idx_tok]
		
		if token:
			return [token]
		else:
			# no se ha terminado de añadir segmentos -> no fin -> no eos
			return self.vocab[:self.eos] + self.vocab[self.eos+1:]
	
	def get_state(self,input_ids):
		cur_seg = 0
		last_pos = 0
		i = 0
		while cur_seg < len(self.segments) and i <= len(input_ids) - len(self.segments[cur_seg]):
			if input_ids[i:(i+len(self.segments[cur_seg]))].tolist() == self.segments[cur_seg]:
				i += len(self.segments[cur_seg])
				last_pos = i
				cur_seg += 1
			else:
				i += 1
		if cur_seg > 0 and last_pos == len(input_ids):
			return cur_seg-1, len(self.segments[cur_seg-1]), last_pos
		if cur_seg < len(self.segments):
			#if cur_seg > 0 and input_ids[last_pos:].tolist() == self.segments[cur_seg-1]:
			cur_tok = min(len(input_ids),len(self.segments[cur_seg]))
			while cur_tok > 0 and input_ids[-cur_tok:].tolist() != self.segments[cur_seg][:cur_tok]:
				cur_tok -= 1
			if cur_tok == 0:
				cur_tok = -1
			return cur_seg, cur_tok, last_pos
		else:
			return cur_seg, -1, last_pos
	

def argmax_last(arr):
	arr = arr.flatten()
	max_value = np.max(arr)
	max_indices = np.where(arr == max_value)[0]
	return max_indices[-1]

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
	if args.final > -1:
		src_lines = read_file(file_name)[:args.final]
	else:
		src_lines = read_file(file_name)
	file_name = '{0}/test.{1}'.format(args.folder, args.target)
	if args.final > -1:
		trg_lines = read_file(file_name)[:args.final]
	else:
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
	#|=========================================================s	
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
			'''aux = tokenize(c_trg)
			aux = ' '.join([aux[idx] + '(' + str(idx) + ')' for idx in range(len(aux))])
			print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,c_src,aux))'''
			print("Sentece {0}:\n\tSOURCE: {1}\n\tTARGET: {2}".format(i+1,c_src,c_trg))

		ite = 0
		MAX_TOKENS = 128
		restrictor = Restrictor(VOCAB,tokenizer,len(tokenize(c_trg)),filters=args.filters,values=args.values)
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
				#constraints = restrictor.create_constraints(filters=[], values=[],verbose=args.verbose)
				restrictor.prepare()

				#print('generando')
				raw_output = model.generate(**encoded_src,
								forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code],
								max_new_tokens=MAX_TOKENS,
								prefix_allowed_tokens_fn=restrictor.restrict)
				generated_tokens = raw_output.tolist()[0]
				if len(generated_tokens) >= MAX_TOKENS:
					MAX_TOKENS = min(512, int(MAX_TOKENS*(3/2)))
				elif len(generated_tokens) > 3/4 * MAX_TOKENS:
					MAX_TOKENS = min(512, int(MAX_TOKENS*(5/4)))
				output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

				if args.verbose:
					'''aux = tokenize(output)
					aux = ' '.join([aux[idx] + '(' + str(idx) + ')' for idx in range(len(aux))])
					print("ITE {0} ({1}): {2}".format(ite, len(generated_tokens), aux))'''
					print("ITE_TOK({0}): {1}".format(len(generated_tokens),generated_tokens))
					print("ITE {0} ({1}): {2}".format(ite, len(generated_tokens), output))
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
	parser.add_argument("-fin","--final",required=False, default=-1,type=int,help="Final Line")
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
