"""
Machine Translation with Mbart

Example of use:
	> python3 mbart.py -src es -trg en -dir es-en
"""
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import argparse
import torch
import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def translate(args):
	#|========================================================
	#| READ SOURCE DATASET
	file_name = '/dataset/europarl-inmt/{0}/test.{1}'.format(args.folder, args.source)
	file_src = open(file_name, 'r')
	lines = file_src.read().splitlines()
	file_src.close()
	#| PREPARE DOCUMENT TO WRITE
	file_name = '/dataset/europarl-inmt/{0}/mbart.{1}.{2}'.format(args.folder, args.source+args.target,args.target)
	file_out = open(file_name, 'w')
	#|========================================================
	#| LOAD MODEL AND TOKENIZER
	model = MBartForConditionalGeneration.from_pretrained(args.model).to(device)
	tokenizer = MBart50TokenizerFast.from_pretrained('./mbart-large-50-many-to-many-mmt')
	tokenizer.src_lang = args.source_code
	#|========================================================

	for i in range(0, len(lines)):
		article_es = lines[i]
		encoded_es = tokenizer(article_es, return_tensors="pt").to(device)
		generated_tokens = model.generate(**encoded_es, forced_bos_token_id=tokenizer.lang_code_to_id[args.target_code])
		output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
		file_out.write("{}\n".format(output[0]))
		print("Sentence {0}:\n{1}\n{2}\n\n".format(i+1,article_es,output[0]))
	file_out.close()

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
	parser.add_argument("-src", "--source",  required=True,  help="Source Language")
	parser.add_argument("-trg", "--target",  required=True,  help="Target Language")
	parser.add_argument("-dir", "--folder",  required=True,  help="Folder where is the dataset")
	parser.add_argument("-model", "--model", required=False, help="Folder where is the model")

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
