import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
from iso639 import Lang
import os

BAR_WIDTH = 0.2
iso639 = {'en':'Inglés','fr':'Francés','de':'Alemán','es':'Español','gl':'Gallego','sw':'Suajili','ne':'Nepalí'}

def get_latex(x):
    if pd.isna(x):
        return ''
    elif isinstance(x, (int, float, np.number)):
        return f'${x*100:.1f}$'
    else:
        return str(x)

def plot_time(data, filename):
    plt.figure()
    plt.title('Tiempo de ejecución')
    xlabels = data['modelo'].unique()
    y = [min(data[data['modelo'] == modelo]['tiempo']) for modelo in xlabels]
    plt.bar(xlabels,y)
    plt.xticks(xlabels)
    plt.ylabel('Tiempo (s)')
    plt.savefig(filename)

def plot_chart(data,filename, langs):
    MODELS = data['modelo'].unique()
    x = np.arange(len(MODELS))
    pairs = [(src,'en') for src in langs] + [('en',trg) for trg in langs]

    plt.figure(figsize=(17,10))
    for pair, i in zip(pairs, np.arange(1,7)):
        src, trg = pair
        plt.subplot(2,len(pairs)//2,i)
        pref_wsr = [data[(data['modelo'] == modelo) & (data['metodo'] == 'prefix') & 
                      (data['src']==src) & (data['trg']==trg)]['wsr'].values.item() for modelo in MODELS]
        seg_wsr = [data[(data['modelo'] == modelo) & (data['metodo'] == 'segment') & 
                      (data['src']==src) & (data['trg']==trg)]['wsr'].values.item() for modelo in MODELS]
        pref_mar = [data[(data['modelo'] == modelo) & (data['metodo'] == 'prefix') & 
                      (data['src']==src) & (data['trg']==trg)]['mar'].values.item() for modelo in MODELS]
        seg_mar = [data[(data['modelo'] == modelo) & (data['metodo'] == 'segment') & 
                      (data['src']==src) & (data['trg']==trg)]['mar'].values.item() for modelo in MODELS]
        plt.bar(x-3*BAR_WIDTH/2,pref_wsr, BAR_WIDTH, label='WSR prefijos')
        plt.bar(x+BAR_WIDTH-3*BAR_WIDTH/2,seg_wsr, BAR_WIDTH, label='WSR segmentos')
        plt.bar(x+2*BAR_WIDTH-3*BAR_WIDTH/2,pref_mar, BAR_WIDTH, label='MAR prefijos')
        plt.bar(x+3*BAR_WIDTH-3*BAR_WIDTH/2,seg_mar, BAR_WIDTH, label='MAR segmentos')
        plt.xticks(x, MODELS)
        plt.xlabel('Modelo')
        plt.legend()
        plt.title(f'{iso639[src]}-{iso639[trg]}')
    plt.savefig(filename)
    plt.show()

parser = ArgumentParser()
parser.add_argument('file', type=str, default='test.csv', help='Archivo que leer')
parser.add_argument('-o','--output', type=str, default='figuras', help='Directorio donde guardar las tablas y figuras')
parser.add_argument('-opt','--opcion', type=str, default='general', choices=['general','calidad','prefijos','segmentos','tiempos','prefseg'],help='Opción de tabla a mostrar')
parser.add_argument('-l','--lang', default=['fr','de','es','gl','sw','ne'], nargs='+', help='Idiomas a mostrar (a parte del ingles)')
parser.add_argument('-m','--modelo', type=str, help='Modelo a mostrar')
args = parser.parse_args()

data = pd.read_csv(args.file)
data = data[(data['src'].isin(args.lang)) | (data['trg'].isin(args.lang))]
data = data.sort_values(by=['modelo','src','trg'])  

if args.opcion == 'tiempos':
    plot_time(data,os.path.join(args.output,'tiempos.png'))
    exit()
if args.opcion == 'prefseg':
    plot_chart(data,os.path.join(args.output,'pref_seg.png'), args.lang)
    exit()
if args.modelo:
    data = data[data['modelo'] == args.modelo]
    data = data.drop(columns='modelo')

if args.opcion == 'general':
    data = data.drop(columns=['observaciones'])
elif args.opcion == 'calidad':
    data = data[data['metodo'] == 'segment']
    data = data.drop(columns=['metodo','wsr','mar','observaciones'])
elif args.opcion == 'prefijos':
    data = data[data['metodo'] == 'prefix']
    data = data.drop(columns=['metodo','bleu','ter','observaciones'])
else:
    data = data[data['metodo'] == 'segment']
    data = data.drop(columns=['metodo','bleu','ter','observaciones'])

if not os.path.exists(args.output):
    os.makedirs(args.output)
latex_table = data.style.format(get_latex).hide(axis='index').to_latex()
with open(os.path.join(args.output,f'{args.opcion}.tex'), 'w') as f:
    f.write(latex_table)
print(latex_table)