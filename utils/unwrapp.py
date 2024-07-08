from train2 import *

def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the checkpoint")
    parser.add_argument("-src", "--source", required=True, help="Source Language")
    parser.add_argument("-trg", "--target", required=True, help="Target Language")
    parser.add_argument('-model','--model_name',default='mbart',choices=['mbart','m2m','flant5','mt5','llama3','nllb','bloom'],help='Model to train')
    parser.add_argument('-lora','--lora',action='store_true',help='Whether to use LowRank or not')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = read_parameters()
    args = check_parameters(args)

    MODEL = load_model(args.model_name)
    TOKENIZER = load_tokenizer(args)

    if args.lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules='all-linear'
        )
        MODEL = get_peft_model(MODEL, lora_config)

    translator = TranslationModel.load_from_checkpoint(args.path, model=MODEL, tokenizer=TOKENIZER)

    name = f'{args.model_name}_{args.source + args.target}'
    translator.model.save_pretrained(name)
    print(f'Model saved as {name}')