import numpy as np
import sys
from abc import ABC, abstractmethod
from nltk.tokenize.treebank import TreebankWordTokenizer
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          M2M100ForConditionalGeneration, M2M100Tokenizer,
                          MBart50TokenizerFast, MBartForConditionalGeneration,
						  MT5ForConditionalGeneration,
						  BitsAndBytesConfig)

class Restrictor(ABC):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.start_char = '▁'
        self.start_toks = [value for key,value in tokenizer.get_vocab().items() if key[0] == self.start_char]
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.get_vocab()[tokenizer.eos_token]

    @abstractmethod
    def check_segments(self,tgt,hyp,verbose=False):
        '''
            Comprueba los segmentos coincidentes entre la hipotesis y la referencia.
		
		Parameters:
            tgt (str): Referencia.
            hyp (str): Hipotesis.
            verbose (bool): Muestra informacion adicional.
		
		Returns:
            int: Numero de acciones de raton.
            int: Numero de correcciones.
            bool: ¿Se ha terminado de añadir la referencia

		'''
        pass
	
    @abstractmethod
    def prepare(self, model_name):
        '''
		    Prepara el modulo para restringir la generacion de la respuesta.
			
        Parameters:
            model_name (str): Nombre del modelo.
		'''
        pass

    @abstractmethod
    def restrict(self,batch_id, input_ids):
        '''
            Restring un paso de la generacion de la respuesta.
			
        Parameters:
            batch_id (int): Identificador de la muestra.
            input_ids (list): Tokens generados hasta el momento.
		'''
        pass

    @abstractmethod
    def decode(self,input_ids):
        '''
            Convierte los tokens generados a texto
			
        Parameters:
            input_ids (list): Tokens generados.
		
        Returns:
            str: Texto generado.
		'''
        pass

class PrefixRestrictor(Restrictor):
    def __init__(self, vocab, tokenizer, **kwargs):
        super().__init__(vocab,tokenizer)
        self.prefix = ''
        self.tok_prefix = []

    def check_segments(self, target, hyp, verbose = False):
        prefix = []
        correction = 0
        mouse_actions = 0
        word_strokes = 0

        target = tokenize(target)
        hyp = tokenize(hyp)

        i = 0
        while i < len(target):
		#for i in range(len(target)):
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
            i+=1
        prefix = ' '.join(prefix)
        prefix += ' '
        if correction == 0:
            if len(prefix) != len(self.prefix)+1:
                mouse_actions = 1
        elif correction == 1:
            if len(prefix) != len(self.prefix) +1:
                mouse_actions = 1
            word_strokes = 1
        elif correction == 2:
            if len(prefix) != len(self.prefix) +1:
                mouse_actions = 2
            word_strokes = 2
        self.prefix = prefix
        if verbose:
            print('Prefix:', prefix)
            print('Correction:', correction)
        return mouse_actions, word_strokes, i >= len(target)
	
    def prepare(self, model_name):
        self.tok_prefix = self.tokenizer.encode(self.prefix)[:-1]
	
    def restrict(self,batch_idx, prefix_beam):
        pos = len(prefix_beam)
        if pos<len(self.tok_prefix):
            return [self.tok_prefix[pos-1]]
        elif pos==len(self.tok_prefix):
            return self.start_toks
        return self.vocab
	
    def decode(self,input_ids):
        return self.prefix + self.tokenizer.decode(input_ids[len(self.tok_prefix):], skip_special_tokens=True)

class SegmentRestrictor(Restrictor):
	def __init__(self,vocab, tokenizer, target_len, wait_tokens = 3, **kwargs):
		super().__init__(vocab, tokenizer)
		if self.eos in self.start_toks:
			self.start_toks.remove(self.eos)
		# segmentos
		self.prefix = []
		self.correction = ''
		self.segments = []
		self.prev_tseg = -np.ones(target_len+1,dtype=int)
		self.prev_ini = np.zeros(target_len+1,dtype=int)
		# forzado de busqueda
		self.max_wait = wait_tokens
		self.prefix = False

	def check_segments(self,tgt,hyp,verbose=False):
		# tokenizar strings
		tgt = tokenize(tgt)
		hyp = tokenize(hyp)

		lent, lenh = len(tgt), len(hyp)
		dp = cruce(tgt,hyp,lent,lenh)
		t_seg = self.get_segments(dp)
		ini_seg = np.zeros(t_seg.shape[0],dtype=int)
		if verbose: 
			print('Validado:',self.prev_tseg)
			print('Nuevo:   ',t_seg)

		# Primer fallo
		corr = 0
		while corr < lent and t_seg[corr] != -1:
			corr += 1
		i = corr + 1
		# Segundo fallo
		while i < lent and t_seg[i] != -1:
			i += 1

		# SEGMENTOS
		self.segments = ['']
		num_corrections = 0
		# Si la primera palabra validada es el inicio de tgt y no esta a principio de hyp
		# accion de union con el inicio (~prefijo)
		if t_seg[0] > 0:
			mouse_actions = 2
			self.prefix = True
		else:
			mouse_actions = 0
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
		pos = np.sum(ini_seg[:corr])
		self.correction = ''
		if corr < lent:
			self.correction += tgt[corr] + ' '
			t_seg[corr] = lenh
			ini_seg[corr] = 1
			corr += 1
			num_corrections += 1
			mouse_actions += 1
			self.segments = self.segments[:pos] + [self.correction] + self.segments[pos:]
		mouse_actions += 1 # fin de correccion

		self.prev_tseg = t_seg
		self.prev_ini = ini_seg
		if i >= lent:
			mouse_actions = num_corrections
		if verbose:
			print('inicios:',self.prev_ini)
			print('Segmentos:', self.segments)
			print('Correccion:', self.correction)
		return mouse_actions, num_corrections, i >= lent
	
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
				t_seg = lcs(dp,t_seg,last_seg,i-length+1,pos-length,pos)
				t_seg[i-length:i] = np.arange(pos-length,pos)
				last_seg = i
				length = 0
			length += self.prev_tseg[i] != -1
		t_seg = lcs(dp, t_seg, last_seg, dp.shape[0],max(0,t_seg[last_seg]), dp.shape[1])
		return t_seg
	
	def prepare(self, model_name):
		if 't5' in model_name:
			self.tok_segments = [self.tokenizer.encode(s)[:-1] for s in self.segments]
		else:
			self.tok_segments = [self.tokenizer.encode(s)[1:-1] for s in self.segments]
	
	def restrict(self,batch_id, input_ids):
		idx_seg, idx_tok,last_match = self.get_state(input_ids)
		waiting = len(input_ids) - last_match
		# ¿Se ha terminado de añadir segmentos?
		if idx_seg >= len(self.tok_segments):
			return self.vocab	
		
		# ¿hemos terminado de añadir el segmento actual?
		# ultimo token de segmento -> tokens de inicio de palabra
		if idx_tok >= len(self.tok_segments[idx_seg]):
			if idx_seg >= len(self.tok_segments)-1:
				# ultimo token añadido
				return self.start_toks + [self.eos]
			return self.start_toks
		
		token = None
		# ¿prefijo?
		if self.prefix and len(input_ids) == 2:
			#print('prefijo:',self.tok_segments[0][0])
			token = self.tok_segments[0][0]
		# Si no estamos añadiendo ningun segmento miramos si podemos empezar con uno
		elif idx_tok == -1:
			if waiting >= self.max_wait or input_ids[-1] == self.tok_segments[idx_seg][0]:
				# ¿el token sugerido es el inicio del siguiente segmento?
				token = self.tok_segments[idx_seg][0]
		else:
			token = self.tok_segments[idx_seg][idx_tok]
		
		if token:
			#print('token:',token)
			return [token]
		else:
			# no se ha terminado de añadir segmentos -> no fin -> no eos
			return self.vocab[:self.eos] + self.vocab[self.eos+1:]
	
	def get_state(self,input_ids):
		cur_seg = 0
		last_pos = 0
		i = 0
		while cur_seg < len(self.tok_segments) and i <= len(input_ids) - len(self.tok_segments[cur_seg]):
			if input_ids[i:(i+len(self.tok_segments[cur_seg]))].tolist() == self.tok_segments[cur_seg]:
				i += len(self.tok_segments[cur_seg])
				last_pos = i
				cur_seg += 1
			else:
				i += 1
		if cur_seg > 0 and last_pos == len(input_ids):
			return cur_seg-1, len(self.tok_segments[cur_seg-1]), last_pos
		if cur_seg < len(self.tok_segments):
			cur_tok = min(len(input_ids),len(self.tok_segments[cur_seg]))
			while cur_tok > 0 and input_ids[-cur_tok:].tolist() != self.tok_segments[cur_seg][:cur_tok]:
				cur_tok -= 1
			if cur_tok == 0:
				cur_tok = -1
			return cur_seg, cur_tok, last_pos
		else:
			return cur_seg, -1, last_pos

	def decode(self,input_ids):
			idx_seg = 0 # indice de segmentos
			texto = ''
			begin = 0
			tok = 0
			lengths = [len(seg) for seg in self.tok_segments]
			for tok in range(len(input_ids)):
				if idx_seg < len(self.tok_segments) and input_ids[(tok-lengths[idx_seg]):tok] == self.tok_segments[idx_seg]:
					texto += self.tokenizer.decode(input_ids[begin:tok-lengths[idx_seg]]) + ' '
					texto += self.segments[idx_seg] + ' '
					begin = tok + 1
					idx_seg += 1
			if begin < len(input_ids):
				texto += self.tokenizer.decode(input_ids[begin:], skip_special_tokens=True)
			return texto
	
def cruce(tgt,hyp,lent,lenh):
    dp = np.zeros((lent+1,lenh+1),dtype=int)
    for i in range(0,lent):
        for j in range(0,lenh):
            if tgt[i] == hyp[j]:
                dp[i+1,j+1] = dp[i,j] + 1
    return dp

def lcs(dp, tgt, ini_t, fin_t, ini_h, fin_h):
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
        tgt = lcs(dp,tgt,ini_t,ini_lcs[0]+1,ini_h,ini_lcs[1]+1)
    # segmentos posteriores
    if fin_lcs[0] < fin_t:
        tgt = lcs(dp,tgt,fin_lcs[0]+1,fin_t,fin_lcs[1]+1,fin_h)
    return tgt

def tokenize(sentence, wordTokenizer = TreebankWordTokenizer()):
	sentence = sentence.replace('…', '...')
	sentence = sentence.replace('´', '\'')
	sentence = sentence.replace('\'', ' \' ')
	sentence = sentence.replace('.', ' . ')
	sentence = sentence.replace(',', ' , ')
	sentence = sentence.replace('-', ' - ')
	tokens = wordTokenizer.tokenize(sentence)
	for idx, t in enumerate(tokens):
		t = t.replace('``', '"')
		tokens[idx] = t
	return tokens

def detokenize(sentence, wordTokenizer = TreebankWordTokenizer()):
	sentence = sentence.replace(' \' ', '\'')
	sentence = sentence.replace(' . ', '.')
	sentence = sentence.replace(' , ', ',')
	sentence = sentence.replace(' - ', '-')
	tokens = wordTokenizer.tokenize(sentence)
	return tokens

def load_model(model_path, args, _dev=None):
	kwargs = {}
	if args.quantize:
		kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True,device=_dev)
	if args.model_name == 'mbart':
		_mdl = MBartForConditionalGeneration.from_pretrained(model_path, **kwargs)
		_tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", 
											  src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'm2m':
		_mdl = M2M100ForConditionalGeneration.from_pretrained(model_path, **kwargs)
		_tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", 
										 src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'flant5':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("google/flan-t5-base",
									   src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'mt5':
		_mdl = MT5ForConditionalGeneration.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("google/mt5-small",
									   src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'nllb':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
		_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
											src_lang=args.source_code, tgt_lang=args.target_code)
	elif args.model_name == 'bloom':
		_mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path)
		_tok = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
	else:
		print('Model not implemented: {0}'.format(args.model_name))
		sys.exit(1)
	if not args.quantize:
		_mdl.to(_dev)
	return _mdl, _tok

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