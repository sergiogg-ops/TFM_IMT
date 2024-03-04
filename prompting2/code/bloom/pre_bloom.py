from transformers import BloomTokenizerFast, BloomForCausalLM
from nltk.tokenize import wordpunct_tokenize

def read_file(name):
	file_r = open(name, 'r')
	lines = file_r.read().splitlines()
	file_r.close()
	return lines

def check_prefix(target, hyp):
	correction = False
	target = wordpunct_tokenize(target)
	hyp = wordpunct_tokenize(hyp)
	prefix = []

	for i in range(len(target)):
		if len(hyp)<=i:
			correction = True
			prefix.append(target[i])
			break
		elif target[i] == hyp[i]:
			prefix.append(target[i])
		else:
			correction = True
			prefix.append(target[i])
			break
	return ' '.join(prefix), correction

#|======================
#|   READ THE PROMPT
#| >header_prompt
#| >header_size
#|======================
file_r = open('prompt1.txt', 'r')
header_prompt = file_r.read()
header_size = len(header_prompt.splitlines())
file_r .close()

#|====================== 
#|    SOURCE FILE
#| >src_lines
#|====================== 
src_lines = read_file('/dataset/europarl-tst/es-en/tst.es')

#|====================== 
#|    TARGET FILE
#| >trg_lines
#|====================== 
trg_lines = read_file('/dataset/europarl-tst/es-en/tst.en')

#|====================== 
#|    OUTPUT FILE
#| >file_w
#|======================
file_w = open('/dataset/europarl-tst/es-en/bloom.en', 'w')

#|====================== 
#|    LOAD MODELS
#| >tokenizer
#| >model
#|======================
tokenizer = BloomTokenizerFast.from_pretrained("./bloom-1b7", device_map="auto")
model = BloomForCausalLM.from_pretrained("./bloom-1b7", device_map="auto")

total_words = 0
total_ws = 0
total_ma = 0
for i in range(0, len(src_lines)):
	mouse_actions = 0
	word_strokes = 0
	n_words = len(wordpunct_tokenize(trg_lines[i]))
	prefix = ""
	ite = 0

	source = ' '.join(wordpunct_tokenize(src_lines[i]))
	target = ' '.join(wordpunct_tokenize(trg_lines[i]))
	print("Sentence {0}:\n\tSOURCE:{1}\n\tTARGET:{2}\n".format(i+1, source, target))

	while(prefix[0:len(target)] != target):
		prompt = header_prompt + "{0} ::: {1}".format(src_lines[i], prefix)

		input_ids = tokenizer(prompt, return_tensors="pt").input_ids
		generated_tokens = model.generate(input_ids, max_new_tokens=100)
		output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

		if output != '':
			output = output.splitlines()[header_size].split(':::')[1].lstrip(' ')
			output = output.lower()
		print("ITE {0}: {1}".format(ite, output))

		ite += 1
		prefix, correction = check_prefix(target, output)
		prefix = prefix+' 'if prefix[-1]!=' ' else prefix

		if correction:
			mouse_actions += 1
			word_strokes += 1
		else:
			mouse_actions += 1
	print("WSR: {0:.4f} MAR: {1:.4f}".format(word_strokes/n_words, mouse_actions/n_words))
	total_ma += mouse_actions
	total_ws += word_strokes
	total_words += n_words

	if (i+1)%10 == 0:
		print("T_WSR: {0:.4f} T_MAR: {1:.4f}".format(total_ws/total_words, total_ma/total_words))
	print("\n")

	file_w.write("{2} T_WSR: {0:.4f} T_MAR: {1:.4f}".format(total_ws/total_words, total_ma/total_words, i))
