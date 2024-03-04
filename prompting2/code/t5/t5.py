from transformers import T5Tokenizer, MT5ForConditionalGeneration, T5ForConditionalGeneration

file_r = open('/dataset/europarl-inmt/fr-en/test.en', 'r')
lines = file_r.read().splitlines()
file_r.close()

file_w = open('/dataset/europarl-inmt/fr-en/t5.fr', 'w')

tokenizer = T5Tokenizer.from_pretrained("./t5-small", device_map="auto")
model = T5ForConditionalGeneration.from_pretrained("./t5-small", device_map="auto")

for i in range(0, len(lines)):
	article_de = "translate from english to french: {0}".format(lines[i])
	input_ids = tokenizer(article_de, return_tensors="pt").input_ids
	generated_tokens = model.generate(input_ids)
	output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
	file_w.write('{}\n'.format(output))
	print("Sentence {0}:\n{1}\n{2}\n\n".format(i+1,article_de,output))
file_w.close()
