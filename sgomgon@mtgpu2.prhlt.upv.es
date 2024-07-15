import subprocess
from sys import argv

if len(argv) != 3:
    print('Usage: python trace.py <script> <models folder>')
    exit()
script = argv[1]
model_dir = argv[2]

MODELS = ['mbart','m2m','flant5','nllb']
langs = [('fr','en'),('de','en'),('es','en'),('gl','en'),('sw','en')]
datasets = ['europarl-inmt','europarl-inmt','europarl-inmt','hplt','hplt']

for ls, data in zip(langs,datasets):
    for model in MODELS:
        subprocess.run(f'python {script} -model {model_dir}/{model}_{ls[0]+ls[1]} -src {ls[0]} -trg {ls[1]} -dir {data}/{ls[0]}-{ls[1]} -model_name {model} -v -fin 100', shell=True)
        subprocess.run(f'python {script} -model {model_dir}/{model}_{ls[1]+ls[0]} -src {ls[1]} -trg {ls[0]} -dir {data}/{ls[1]}-{ls[0]} -model_name {model} -v -fin 100', shell=True)