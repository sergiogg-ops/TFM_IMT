# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX = 3

with open('europarl-inmt/fr-en/test.fr', 'r') as file_r:
    lines = file_r.read().splitlines()

tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-7B-v0.1")

for i in range(len(lines[:MAX])):
    original = lines[i]
    sentence = 'Translate this from french to english: ' + original
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    generated_tokens = model.generate(input_ids, max_new_tokens=100)
    output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    output = output.split('\n')[-1]
    print(f'SENTENCE {i+1}\n\tORIGINAL: {original}\n\tTRADUCCION: {output}')