from train import *
from model import load_model

def read_parameters():
    parser = argparse.ArgumentParser(description='Unwrapp a model from the lightning checkpoint')
    parser.add_argument("path", help="Path to the checkpoint")
    parser.add_argument("-src", "--source", required=True, help="Source Language")
    parser.add_argument("-trg", "--target", required=True, help="Target Language")
    parser.add_argument('-model','--model_name',default='mbart',choices=['mbart','m2m','flant5','mt5','llama3','nllb','bloom', 'eurollm','gemma'],help='Model contained in the checkpoint')
    parser.add_argument('-lora','--lora',action='store_true',help='Whether LoRA has been used or not')

    args = parser.parse_args()
    return args

def get_path(model_name):
    '''
    Returns the url of the model to download
    Parameters:
        model_name (str): Name of the model to download

    Returns:
        str: URL of the model
    '''
    if model_name == 'mbart':
        return 'facebook/mbart-large-50-many-to-many-mmt'
    elif model_name == 'm2m':
        return "facebook/m2m100_418M"
    elif model_name == 'flant5':
        return "google/flan-t5-base"
    elif model_name == 'nllb':
        return "facebook/nllb-200-distilled-600M"
    elif model_name == 'llama3':
        return "meta-llama/Llama-3.2-1B-Instruct"
    elif model_name == 'eurollm':
        return "utter-project/EuroLLM-1.7B"
    elif model_name == 'gemma':
        return "google/gemma-3-4b-it"
    else:
        print('Model not implemented: {0}'.format(model_name))
        sys.exit(1)

if __name__ == '__main__':
    args = read_parameters()
    args = check_parameters(args)

    MODEL, TOKENIZER = load_model(get_path(args.model_name), args)

    if args.lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules='all-linear'
        )
        MODEL = get_peft_model(MODEL, lora_config)

    translator = TranslationModel.load_from_checkpoint(args.path, 
                                                       map_location='cpu',
                                                       model=MODEL, 
                                                       tokenizer=TOKENIZER)

    name = f'{args.model_name}_{args.source + args.target}'
    translator.model.save_pretrained(name)
    print(f'Model saved as {name}')
