from art import aggregators
from art import scores
from art import significance_tests
from argparse import ArgumentParser
import os
from tqdm import tqdm
import pandas as pd

def read_args():
    parser = ArgumentParser(description='Approximate Randomization Test')
    parser.add_argument('directory', type=str, help='Directory of the scores files')
    parser.add_argument('-r', '--repetitions', type=int, help='Total trials to compute', default=10000)
    parser.add_argument('-pv', '--pvalue', type=float, help='P-value to consider', default=0.05)
    parser.add_argument('-pair', '--pair', required=True, help='Pair of languages to compare')
    parser.add_argument('-m', '--models', default=['mbart','m2m','flant5','nllb'], nargs='+', help='Models to compare')
    parser.add_argument('-sys','--system', default='sb', choices=['sb','pb'], help='System to compare')
    parser.add_argument('-a','--append', action='store_true', help='Append to the output file')
    parser.add_argument('-o', '--output', required=True, type=str, help='Output file')
    return parser.parse_args()

def read_file(directory, mode, model, lang):
    wsr = []
    mar = []
    filename = model + '.' + lang
    #filename = mode + '_imt_' + model + '.' + lang
    if filename in os.listdir(directory):
        filename = os.path.join(directory, filename)
        with open(filename,'r') as f:
            for line in f.readlines()[1:-1]:
                line = line.split('\t')
                wsr.append(scores.Score([float(line[0])]))
                mar.append(scores.Score([float(line[1])]))
        return scores.Scores(wsr), scores.Scores(mar)
    else:
        return False, False

def main():
    args = read_args()
    #significance = pd.DataFrame(columns=['src','trg','model1','model2','sig_wsr','sig_mar'])
    significance = pd.DataFrame(columns=['src','trg','model1','model2','sig_bleu','sig_ter'])
    enumeration = [(args.models[i],args.models[j]) for i in range(len(args.models)) for j in range(i+1,len(args.models))]
    for model1, model2 in tqdm(enumeration, desc='Comprobando significatividad'):
        m1_wsr, m1_mar = read_file(args.directory,args.system,model1,args.pair[2:])
        m2_wsr, m2_mar = read_file(args.directory,args.system,model2,args.pair[2:])
        if all([m1_mar, m1_wsr, m2_mar, m2_wsr]):
            sig_wsr = significance_tests.ApproximateRandomizationTest(m1_wsr, m2_wsr, aggregators.average, trials=args.repetitions).run()
            sig_mar = significance_tests.ApproximateRandomizationTest(m1_mar, m2_mar, aggregators.average, trials=args.repetitions).run()
            #significance = significance._append({'src': args.pair[:2], 'trg': args.pair[2:], 'model1':model1, 'model2':model2,
            #                                        'sig_wsr': sig_wsr<args.pvalue, 'sig_mar': sig_mar<args.pvalue}, ignore_index=True)
            significance = significance._append({'src': args.pair[:2], 'trg': args.pair[2:], 'model1':model1, 'model2':model2,
                                                    'sig_bleu': sig_wsr<args.pvalue, 'sig_ter': sig_mar<args.pvalue}, ignore_index=True)
    significance.to_csv(args.output,mode='a' if args.append else 'w',index=False,header=not args.append)

if __name__ == '__main__':
    main()