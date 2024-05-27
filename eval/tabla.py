import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from argparse import ArgumentParser

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

parser = ArgumentParser()
parser.add_argument('-f','--file', type=str, default='test.csv', help='Archivo que leer')
parser.add_argument('-o','--output', type=str, default='tabla.txt', help='Archivo donde guardar la tabla')
parser.add_argument('-p','--plot', action='store_true', help='Mostrar tabla en gráficos')
parser.add_argument('-pref','--prefix', action='store_true', help='Mostrar tabla de prefijos')
parser.add_argument('-opt','--opcion', type=str, default='general', choices=['general','bleu_ter','wsr_mar'],help='Opción de tabla a mostrar')
parser.add_argument('-m','--modelo', type=str, help='Modelo a mostrar')
args = parser.parse_args()

data = pd.read_csv(args.file)
if args.modelo:
    data = data[data['modelo'] == args.modelo]
if args.opcion == 'general':
    data = data.drop(columns=['modelo','observaciones'])
elif args.opcion == 'bleu_ter':
    data = data.drop(columns=['wsr','mar','observaciones'])
else:
    data = data.drop(columns=['bleu','ter','observaciones'])

if not args.prefix:
    data = data[data['metodo'] != 'prefix']

latex_table = get_latex(data)
with open(args.output, 'w') as f:
    f.write(latex_table)
if args.plot:
    plot_table(data, args.prefix)