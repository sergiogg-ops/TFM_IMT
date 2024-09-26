'''
Generates a chart showing either the BLEU and TER scores or the WSR and MAR scores for a given language pair in both directions.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAR_WIDTH = 0.2
iso639 = {'en':'Inglés','fr':'Francés','de':'Alemán','es':'Español','gl':'Gallego','sw':'Suajili','ne':'Nepalí'}

def bleu_ter(data,filename, langs):
    MODELS = data['modelo'].unique()
    x = np.arange(len(MODELS))
    pairs = [(src,'en') for src in langs] + [('en',trg) for trg in langs]

    plt.figure(figsize=(17,10))
    plt.title('Calidad de las hipótesis de traducción')
    for pair, i in zip(pairs, np.arange(1,7)):
        src, trg = pair
        plt.subplot(2,len(pairs)//2,i)
        bleu = [data[(data['modelo'] == modelo) & (data['metodo'] == 'segment') & 
                      (data['src']==src) & (data['trg']==trg)]['bleu'].values.item()*100 for modelo in MODELS]
        ter = [data[(data['modelo'] == modelo) & (data['metodo'] == 'segment') & 
                      (data['src']==src) & (data['trg']==trg)]['ter'].values.item()*100 for modelo in MODELS]
        plt.bar(x-BAR_WIDTH/2,bleu, BAR_WIDTH, label='BLEU')
        plt.bar(x+BAR_WIDTH-BAR_WIDTH/2,ter, BAR_WIDTH, label='TER')
        if src == 'sw' or trg == 'sw':
            plt.yscale('log')
        #plt.yticks([10,100],[10,100])
        plt.xticks(x, MODELS)
        plt.xlabel('Modelo')
        plt.legend()
        plt.title(f'{iso639[src]}-{iso639[trg]}')
    plt.savefig(filename)
    #plt.show()

def wsr_mar(data,filename, src, trg):
    MODELS = data['modelo'].unique()
    x = np.arange(len(MODELS))

    plt.figure(figsize=(13,7))
    plt.subplot(1,2,1)
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

    plt.subplot(1,2,2)
    pref_wsr = [data[(data['modelo'] == modelo) & (data['metodo'] == 'prefix') & 
                    (data['src']==trg) & (data['trg']==src)]['wsr'].values.item() for modelo in MODELS]
    seg_wsr = [data[(data['modelo'] == modelo) & (data['metodo'] == 'segment') & 
                    (data['src']==trg) & (data['trg']==src)]['wsr'].values.item() for modelo in MODELS]
    pref_mar = [data[(data['modelo'] == modelo) & (data['metodo'] == 'prefix') & 
                    (data['src']==trg) & (data['trg']==src)]['mar'].values.item() for modelo in MODELS]
    seg_mar = [data[(data['modelo'] == modelo) & (data['metodo'] == 'segment') & 
                    (data['src']==trg) & (data['trg']==src)]['mar'].values.item() for modelo in MODELS]
    plt.bar(x-3*BAR_WIDTH/2,pref_wsr, BAR_WIDTH, label='WSR prefijos')
    plt.bar(x+BAR_WIDTH-3*BAR_WIDTH/2,seg_wsr, BAR_WIDTH, label='WSR segmentos')
    plt.bar(x+2*BAR_WIDTH-3*BAR_WIDTH/2,pref_mar, BAR_WIDTH, label='MAR prefijos')
    plt.bar(x+3*BAR_WIDTH-3*BAR_WIDTH/2,seg_mar, BAR_WIDTH, label='MAR segmentos')
    plt.xticks(x, MODELS)
    plt.xlabel('Modelo')
    plt.legend()
    plt.title(f'{iso639[trg]}-{iso639[src]}')
    plt.savefig(filename)

data = pd.read_csv('resultados/test.csv')
filename = 'figuras/bleu_ter.png'
#langs = ['fr','es','de']
langs = ['gl','sw']
bleu_ter(data, filename, langs)
#wsr_mar(data, filename, 'sw', 'en')