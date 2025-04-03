from train import *

def read_parameters():
    parser = argparse.ArgumentParser(description='Unwrapp a model from the lightning checkpoint')
    parser.add_argument("path", help="Path to the checkpoint")
    parser.add_argument("-src", "--source", required=True, help="Source Language")
    parser.add_argument("-trg", "--target", required=True, help="Target Language")
    parser.add_argument('-model','--model_name',default='mbart',choices=NAMES,help='Model contained in the checkpoint')
    parser.add_argument('-lora','--lora',action='store_true',help='Whether LoRA has been used or not')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = read_parameters()
    args = check_parameters(args)

    MODEL, TOKENIZER = load_model(get_url(args.model_name), args)

    if args.lora:
        MODEL = apply_lora(MODEL)

    translator = TranslationModel.load_from_checkpoint(args.path, model=MODEL, tokenizer=TOKENIZER)

    name = f'{args.model_name}_{args.source + args.target}'
    translator.model.save_pretrained(name)
    print(f'Model saved as {name}')