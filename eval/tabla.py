import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from argparse import ArgumentParser
import numpy as np
import os

BAR_WIDTH = 0.2 
XTICKS = [('fr','en'),('en','fr'),('es','en'),('en','es'),('de','en'),('en','de')]

def get_latex(data):
    data_tex = data.copy()
    porcien = lambda x: f'{x*100:.2f}'
    if args.opcion == 'general' or args.opcion == 'bleu_ter':
        data_tex['bleu'] = data_tex['bleu'].apply(porcien)
        data_tex['ter'] = data_tex['ter'].apply(porcien)
    if args.opcion == 'general' or args.opcion == 'wsr_mar':
        data_tex['wsr'] = data_tex['wsr'].apply(porcien)
        data_tex['mar'] = data_tex['mar'].apply(porcien)
    return tabulate(data_tex, headers='keys', tablefmt='latex', showindex=False)

def plot_table(data,show_prefix=True):
    porcentaje = lambda x: f'{x:.2%}'
    if args.opcion == 'general' or args.opcion == 'bleu_ter':
        data['bleu'] = data['bleu'].apply(porcentaje)
        data['ter'] = data['ter'].apply(porcentaje)
    if args.opcion == 'general' or args.opcion == 'wsr_mar':
        data['wsr'] = data['wsr'].apply(porcentaje)
        data['mar'] = data['mar'].apply(porcentaje)

    plt.figure()
    if show_prefix:
        plt.subplot(2,1,1)
        plt.axis('off')
        prefix = data[data['metodo'] == 'prefix'].drop(columns='metodo')
        table = plt.table(cellText=prefix.values, colLabels=prefix.columns, loc='center')
        table.auto_set_font_size(True)
        table.scale(1.3, 1.3)
        plt.title('Prefix-based')

        plt.subplot(2,1,2)
    plt.axis('off')
    segment = data[data['metodo'] == 'segment'].drop(columns='metodo')
    table = plt.table(cellText=segment.values, colLabels=segment.columns, loc='center')
    table.auto_set_font_size(True)
    table.scale(1.3, 1.3)
    plt.title('Segment-based')
    plt.show()

def plot_bleu(data, filename):
    num_series = len(data['modelo'].unique())
    plt.figure()
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['bleu'].values.item() for src,trg in XTICKS]
        serie = [x*100 for x in serie]
        plt.bar(np.arange(6)+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(np.arange(6), ['-'.join(x) for x in XTICKS])
    plt.xlabel('Pares de idiomas')
    plt.ylabel('BLEU (%)')
    plt.legend()
    plt.savefig(filename)

def plot_ter(data,filename):
    num_series = len(data['modelo'].unique())
    plt.figure()
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['ter'].values.item() for src,trg in XTICKS]
        serie = [x*100 for x in serie]
        plt.bar(np.arange(6)+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(np.arange(6), ['-'.join(x) for x in XTICKS])
    plt.xlabel('Pares de idiomas')
    plt.ylabel('TER (%)')
    plt.legend()
    plt.savefig(filename)    

def plot_wsr(data,filename):
    num_series = len(data['modelo'].unique())
    plt.figure()
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['wsr'].values.item() for src,trg in XTICKS]
        serie = [x*100 for x in serie]
        plt.bar(np.arange(6)+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(np.arange(6), ['-'.join(x) for x in XTICKS])
    plt.xlabel('Pares de idiomas')
    plt.ylabel('WSR (%)')
    plt.legend()
    plt.savefig(filename)

def plot_mar(data,filename):
    num_series = len(data['modelo'].unique())
    plt.figure()
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['mar'].values.item() for src,trg in XTICKS]
        serie = [x*100 for x in serie]
        plt.bar(np.arange(6)+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(np.arange(6), ['-'.join(x) for x in XTICKS])
    plt.xlabel('Pares de idiomas')
    plt.ylabel('MAR (%)')
    plt.legend()
    plt.savefig(filename)

def plot_time(data,filename):
    plt.figure()
    plt.title('Tiempo de ejecución')
    xlabels = data['modelo'].unique()
    y = [min(data[data['modelo'] == modelo]['tiempo']) for modelo in xlabels]
    plt.bar(xlabels,y)
    plt.xticks(xlabels)
    plt.ylabel('Tiempo (s)')
    plt.savefig(filename)

def plot_chart(data):
    x = np.arange(6)
    data = data[data['metodo'] != 'prefix']
    num_series = len(data['modelo'].unique())

    plt.figure()
    plt.subplot(2,2,1)
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['bleu'].values.item() for src,trg in XTICKS]
        plt.bar(x+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(x, ['-'.join(x) for x in XTICKS])
    plt.title('BLEU')
    plt.legend()

    plt.subplot(2,2,2)
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['ter'].values.item() for src,trg in XTICKS]
        plt.bar(x+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(x, ['-'.join(x) for x in XTICKS])
    plt.title('TER')
    plt.legend()
    
    plt.subplot(2,2,3)
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['wsr'].values.item() for src,trg in XTICKS]
        plt.bar(x+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(x, ['-'.join(x) for x in XTICKS])
    plt.title('WSR')
    plt.legend()

    plt.subplot(2,2,4)
    for i, modelo in enumerate(data['modelo'].unique()):
        serie = [data[(data['modelo'] == modelo) & (data['src']==src) & (data['trg']==trg)]['mar'].values.item() for src,trg in XTICKS]
        plt.bar(x+BAR_WIDTH*i-(num_series-1)*BAR_WIDTH/2,serie, BAR_WIDTH, label=modelo)
    plt.xticks(x, ['-'.join(x) for x in XTICKS])
    plt.title('MAR')
    plt.legend()
    plt.show()

parser = ArgumentParser()
parser.add_argument('-f','--file', type=str, default='test.csv', help='Archivo que leer')
parser.add_argument('-o','--output', type=str, default='figuras', help='Directorio donde guardar las tablas y figuras')
parser.add_argument('-p','--plot', action='store_true', help='Mostrar tabla en gráficos')
parser.add_argument('-graf','--grafico', action='store_true', help='Mostrar gráficos')
parser.add_argument('-pref','--prefix', action='store_true', help='Mostrar tabla de prefijos')
parser.add_argument('-opt','--opcion', type=str, default='general', choices=['general','bleu_ter','wsr_mar','tiempos'],help='Opción de tabla a mostrar')
parser.add_argument('-m','--modelo', type=str, help='Modelo a mostrar')
args = parser.parse_args()

data = pd.read_csv(args.file)

if args.opcion == 'tiempos':
    plot_time(data,os.path.join(args.output,'tiempos.png'))
    exit()
if args.modelo:
    data = data[data['modelo'] == args.modelo]
    data = data.drop(columns='modelo')
if args.grafico:
    plot_chart(data)

if args.opcion == 'general':
    data = data.drop(columns=['modelo','observaciones'])
elif args.opcion == 'bleu_ter':
    data = data.drop(columns=['wsr','mar','observaciones'])
else:
    data = data.drop(columns=['bleu','ter','observaciones'])

if not args.prefix:
    data = data[data['metodo'] != 'prefix']
    data = data.drop(columns='metodo')

if not os.path.exists(args.output):
    os.makedirs(args.output)
latex_table = get_latex(data)
with open(os.path.join(args.output,'tabla.tex'), 'w') as f:
    f.write(latex_table)
if args.opcion != 'wsr_mar':
    plot_bleu(data,os.path.join(args.output,'bleu.png'))
    plot_ter(data,os.path.join(args.output,'ter.png'))
if args.opcion != 'bleu_ter':
    plot_wsr(data,os.path.join(args.output,'wsr.png'))
    plot_mar(data,os.path.join(args.output,'mar.png'))
if args.plot:
    plot_table(data, args.prefix)