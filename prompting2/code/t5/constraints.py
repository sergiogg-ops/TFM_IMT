from transformers import Constraint
from typing import List, Optional
from abc import ABC, abstractmethod

from transformers import T5TokenizerFast, T5Tokenizer, MT5ForConditionalGeneration, T5ForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from mosestokenizer import MosesTokenizer
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class HF_Fachade(ABC):
	def __init__(self, model_path: str, src_lan: str, trg_lan: str):
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.model_path = model_path
		self.src_lan = src_lan
		self.trg_lan = trg_lan
		self.generate_args = {}
		self.segmentor_args = {}

		self.space_token = None
		self.eos_token = None

		self.model = None

		self.tokenizer = None
		self.start_tokens = None

		self.set_model(self.model_path)
		self.set_tokenizer(self.model_path)

	def encode(self, src: str):
		return self.tokenizer(src,add_special_tokens=False).input_ids

	def translate(self, src: str):
		input_ids = self.tokenizer(src, return_tensors="pt").to(self.device)
		generated_tokens = self.model.generate(
			**input_ids,
			**self.generate_args,
			max_new_tokens=200,
			)
		translation = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
		translation = self.tokenize(translation, self.trg_lan)
		return translation

	def segment_translate(self, src: str, segment_list: List[str]):
		segments_tokenized = [self.tokenizer(text_target=segment + '▁', add_special_tokens=False).input_ids for segment in segment_list]
		segment_ruler = Segmentor(
			len(self.tokenizer),
			segments_tokenized,
			**self.segmentor_args,
			eos_token=self.eos_token,
			start_tokens=self.start_tokens)
		input_ids = self.tokenizer(src, return_tensors="pt").to(self.device)
		generated_tokens = self.model.generate(
			**input_ids,
			**self.generate_args,
			prefix_allowed_tokens_fn=segment_ruler.segment_constraint,
			num_beams=1,
			early_stopping=False,
			max_new_tokens=500)
		translation = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
		translation = self.tokenize(translation, self.trg_lan)
		return translation

	def tokenize(self, sentence, lan):
		sentence = sentence.replace('…', '...')
		sentence = sentence.replace('´', '\'')
		sentence = sentence.replace('\'', ' \' ')
		sentence = sentence.replace('.', ' . ')
		sentence = sentence.replace(',', ' , ')

		sentence = sentence.replace('&lt;', '<')       # MOSES Tokenizer
		sentence = sentence.replace('&gt;', '>')       # MOSES Tokenizer
		sentence = sentence.replace('&quot;', '"')     # MOSES Tokenizer
		sentence = sentence.replace('&amp;', '&')      # MOSES Tokenizer
		sentence = sentence.replace('&apos;', ' \' ')  # MOSES Tokenizer
		with MosesTokenizer(lan) as tokenizer:
			tokens = tokenizer(sentence)
		for idx, t in enumerate(tokens):
			t = t.replace('``', '"')
			t = t.replace('@-@', '-')
			tokens[idx] = t
		return tokens

	@abstractmethod
	def set_model(self, model_path: str):
		raise NotImplementedError(
			f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
		)

	@abstractmethod
	def set_tokenizer(self, model_path: str):
		raise NotImplementedError(
			f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
		)

class MT5_Fachade(HF_Fachade):
	def __init__(self, model_path: str, src_lan: str, trg_lan: str):
		super(MT5_Fachade, self).__init__(model_path, src_lan, trg_lan)

		self.space_token = 259
		self.eos_token = 1
		space_between_tokens = 3
		self.segmentor_args = {
			'wait_tokens': space_between_tokens,
			'wait_offset': space_between_tokens
		}

	def set_model(self, model_path: str):
		self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

	def set_tokenizer(self, model_path: str):
		self.tokenizer = T5TokenizerFast.from_pretrained(model_path)

class MBART_Fachade(HF_Fachade):
	def __init__(self, model_path: str, src_lan: str, trg_lan: str):
		self.codes = {'ar':'ar_AR', 'cs':'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX', 'et': 'et_EE', 'fi': 'fi_FI', 'fr': 'fr_XX', 'gu': 'gu_IN', 'hi': 'hi_IN', 'it': 'it_IT', 'ja': 'ja_XX', 'kk': 'kk_KZ', 'ko': 'ko_KR', 'lt': 'lt_LT', 'lv': 'lv_LV', 'my': 'my_MM', 'ne': 'ne_NP', 'nl': 'nl_XX', 'ro': 'ro_RO', 'ru': 'ru_RU', 'si': 'si_LK', 'tr': 'tr_TR', 'vi': 'vi_VN', 'zh': 'zh_CN', 'af': 'af_ZA', 'az': 'az_AZ', 'bn': 'bn_IN', 'fa': 'fa_IR', 'he': 'he_IL', 'hr': 'hr_HR', 'id': 'id_ID',  'ka': 'ka_GE', 'km': 'km_KH', 'mk': 'mk_MK', 'ml': 'ml_IN', 'mn': 'mn_MN', 'mr': 'mr_IN', 'pl': 'pl_PL', 'ps': 'ps_AF', 'pt': 'pt_XX', 'sv': 'sv_SE', 'sw': 'sw_KE', 'ta': 'ta_IN', 'te': 'te_IN', 'th': 'th_TH', 'tl': 'tl_XX', 'uk': 'uk_UA', 'ur': 'ur_PK', 'xh': 'xh_ZA', 'gl': 'gl_ES', 'sl': 'sl_SI'}
		self.src_code = self.codes[src_lan]
		self.trg_code = self.codes[trg_lan]
		super(MBART_Fachade, self).__init__(model_path, src_lan, trg_lan)

		self.space_token = 5 
		self.eos_token = 2
		self.generate_args = {
			'forced_bos_token_id':self.tokenizer.lang_code_to_id[self.trg_code],
		}
		space_between_tokens = 3
		self.segmentor_args = {
			'wait_tokens': space_between_tokens,
			'wait_offset': space_between_tokens-1
		}

	def set_model(self, model_path: str):
		self.model = MBartForConditionalGeneration.from_pretrained(model_path, device_map="auto")

	def set_tokenizer(self, model_path: str):
		self.tokenizer = MBart50TokenizerFast.from_pretrained(
			model_path, 
			src_lang=self.src_code, 
			tgt_lang=self.trg_code
			)
		vocab = self.tokenizer.get_vocab()
		
		self.start_tokens = []
		for (key, value) in vocab.items():
			if not key[0].isalpha():
				self.start_tokens.append(value)


class Segmentor():
	def __init__(self, vocab_size:int, segments: List[List[int]] = [], wait_tokens: int = 3, wait_offset:int = 0, eos_token:int =2, start_tokens:List[int] =None):
		self.segments = segments
		self.vocab_size = vocab_size
		self.max_wait_time = wait_tokens
		self.reset_segment_cursor()

		self.eos_token = eos_token
		self.tokens_waiting = wait_offset

		self.start_tokens = start_tokens

	def set_segments(self, segments):
		self.segments = segments
		self.reset_segment_cursor()

	def reset_segment_cursor(self):
		self.added_segment = 0
		self.token_position = -1
		self.tokens_waiting = 0

	def segment_constraint(self, batch_id, input_ids):
		# Comprobamos si ya se han anyadido todos los segmentos
		if self.added_segment >= len(self.segments):
			# Checkear Condicion | Queremos asegurar que si es la primera palabra despues de terminar el ultimo segmento empieza un toquen de inicio de palabra
			vocab = [*range(self.vocab_size)] if self.tokens_waiting > 1 else self.start_tokens
			# Para asegurar que solo acurra la primera vez la condicion ya ponemos tokens waiting a un valor dentro de la condicion general
			self.tokens_waiting = 2
			return vocab
		
		# Comprobamos si estamos anyadiendo algun segmento | ==-1 implica que aun no estamos anyadiendo
		if self.token_position == -1:
			# Miramos si alguna de las posibles nuevas palabras es la primera de la del siguiente segmento
			if input_ids[-1] == self.segments[self.added_segment][0]:
				# Al ya aparecer la palabra en posicion 0 ponemos la variable a 1 de forma que a la siguiente ponga la segunda palabra
				self.token_position = 1 
				# Comprobamos si el segmento a anyadir es solamente una palabra
				if len(self.segments[self.added_segment])==1:
					# En caso de serlo el segmento va a ser completado con esta iteracion y debemos preparar para leer el siguiente
					self.tokens_waiting = 0
					self.token_position = -1
					self.added_segment += 1
			# No coincide la palabra, pero ya hemos superado la cantidad maxima de palabras que habiamos fijado entre segmentos validados
			elif self.tokens_waiting >= self.max_wait_time:
				self.token_position = 0

		# Comprobamos si tenemos que poner algun token
		if self.token_position != -1:
			# Preparamos el token a poner y aumentamos la posicion en 1 para la siguiente iteracion
			token = self.segments[self.added_segment][self.token_position]
			self.token_position += 1
			# Comprobamos si ya se han puesto todos los valores del segmento
			if self.token_position >= len(self.segments[self.added_segment]):
				# Si es asi reseteamos variable, y aumentamos added_segment para ir al siguiente
				self.tokens_waiting = 0
				self.token_position = -1
				self.added_segment += 1	
			# Enviamos el token
			return [token]
		# Si no se han puesto ningun token del segmento aumentamos el tiempo de espera que llevamos
		self.tokens_waiting += 1

		# Comprobamos si el tiempo es mayor a 1 o no
		if self.tokens_waiting > 1:
			# Si lo es podemos poner cualquier token menos el de final de oracion | Aun quedan segmentos
			vocab = [*range(self.vocab_size)] 
			vocab.remove(self.eos_token)
		else:
			# Si es el primer token despues de terminar de poner otro segmento pedimos token de inicio de palabra
			vocab = self.start_tokens
		return vocab		


