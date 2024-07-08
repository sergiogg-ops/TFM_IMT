import subprocess
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
import os

print(sys.argv)
if len(sys.argv) != 3:
    print('ERROR: python corpus_table.py vocabulary.sh dataset')
    exit(-1)

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
for file in tqdm([os.path.join(sys.argv[2], f) for f in os.listdir(sys.argv[2])]):
    print(file)
    print('\tS,W')
    subprocess.run(['wc','-lw',file])
    with open(file) as f:
        tok_lines = [tokenizer.convert_ids_to_tokens(line) for line in tokenizer(f.readlines())['input_ids']]
    with open('tok_data.txt','w') as f:
        for line in tok_lines:
            f.write(' '.join(line) + '\n')
    print('\tV')
    with open('vocab.txt','w') as file:
        subprocess.run(['./'+sys.argv[1],'tok_data.txt'],stdout=file)

    subprocess.run(['wc','-l','vocab.txt'])
subprocess.run(['rm','tok_data.txt'])
subprocess.run(['rm','vocab.txt'])