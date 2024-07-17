import numpy as np
import matplotlib.pyplot as plt


sents = {'Entrenamiento': np.array([1900000,1900000,2000000,2000000,2000000,2000000,956800,956800,1700000,1700000]),
        'Validacion': np.array([3000]*10),
        'Test': np.array([2200,2200,3000,3000,1500,1500,3000,3000,3000,3000])}

tokens = {'Entrenamiento': np.array([49800000,52300000,57100000,54500000,60500000,54500000,13100000,12500000,21000000,20000000]),
          'Validacion': np.array([63500,64800,78900,73000,73700,64800,40600,38900,36300,34500]),
          'Test': np.array([44400,46800,70300,64600,29500,26800,41900,40300,36800,34900])}

vocabs = {'Entrenamiento': np.array([15500,25100,14900,25200,16000,25200,22500,28100,25600,28000]),
          'Validacion': np.array([4500,8800,4600,9200,5300,8800,5600,9400,5200,6900]),
          'Test': np.array([4000,7400,4400,8800,3800,5300,5500,9400,5300,7200])}

BAR_WIDTH = 0.25
def plot(dictionary):
    i = 0
    for fold, values in dictionary.items():
        plt.bar(x+BAR_WIDTH*i-(3-1)*BAR_WIDTH/2,values, BAR_WIDTH, label=fold)
        i+=1

x = np.arange(1,11)

plt.figure()
plot(sents)
plt.legend()
plt.xticks(range(1,11),['en-de','de-en','en-es','es-en','en-fr','fr-en','en-gl','gl-en','en-sw','sw-en'])
plt.yscale('log')
plt.title('Número de oraciones')
plt.savefig('figuras/sents.png')

plt.figure()
i=0
plot(tokens)
plt.legend()
plt.xticks(range(1,11),['en-de','de-en','en-es','es-en','en-fr','fr-en','en-gl','gl-en','en-sw','sw-en'])
plt.yscale('log')
plt.title('Número de tokens')
plt.savefig('figuras/tokens.png')

plt.figure()
plot(vocabs)
plt.legend()
plt.xticks(range(1,11),['en-de','de-en','en-es','es-en','en-fr','fr-en','en-gl','gl-en','en-sw','sw-en'])
plt.title('Tamaño del vocabulario')
plt.savefig('figuras/vocabs.png')
