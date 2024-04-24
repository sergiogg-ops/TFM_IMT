import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('eval/metricas.csv')

data = data.drop(columns='observaciones')
data = data.drop(columns='modelo')
porcentaje = lambda x: f'{x:.2%}'
data['bleu'] = data['bleu'].apply(porcentaje)
data['ter'] = data['ter'].apply(porcentaje)
data['wsr'] = data['wsr'].apply(porcentaje)
data['mar'] = data['mar'].apply(porcentaje)

plt.subplot(2,1,1)
plt.axis('off')
prefix = data[data['metodo'] == 'prefix'].drop(columns='metodo')
table = plt.table(cellText=prefix.values, colLabels=prefix.columns, loc='center')
table.auto_set_font_size(True)
#table.set_fontsize(10)
table.scale(1.3, 1.3)
plt.title('Prefix-based')

plt.subplot(2,1,2)
plt.axis('off')
segment = data[data['metodo'] == 'segment'].drop(columns='metodo')
table = plt.table(cellText=segment.values, colLabels=segment.columns, loc='center')
table.auto_set_font_size(True)
#table.set_fontsize(10)
table.scale(1.3, 1.3)
plt.title('Segment-based')
plt.show()