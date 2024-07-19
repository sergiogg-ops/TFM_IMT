import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAR_WIDTH = 0.2
iso639 = {'en':'Inglés','fr':'Francés','de':'Alemán','es':'Español','gl':'Gallego','sw':'Suajili','ne':'Nepalí'}

def plot_chart(data,filename, langs):
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
        #plt.yscale('log')
        #plt.yticks([10,100],[10,100])
        plt.xticks(x, MODELS)
        plt.xlabel('Modelo')
        plt.legend()
        plt.title(f'{iso639[src]}-{iso639[trg]}')
    plt.savefig(filename)
    #plt.show()

data = pd.read_csv('resultados/test.csv')
filename = 'figuras/bleu_ter.png'
langs = ['fr','es','de']
#langs = ['gl','sw']
plot_chart(data, filename, langs)