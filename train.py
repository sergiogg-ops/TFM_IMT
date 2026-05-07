from model import *
import lightning as L
from evaluate import load
import evaluate
import argparse
import torch
import sys
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class TranslationModel(L.LightningModule):
    '''
    Pytorch Lightning wrapper module for the training
    '''
    def __init__(self, model, tokenizer, lr=1e-5, total_steps=10000):
        '''
        Parameters:
            model (torch.nn.Module): Model to train
            tokenizer (Tokenizer): Tokenizer to use
            lr (float): Learning rate of the optimizer
            total_steps (int): Total number of training steps (used for LR scheduler)
        '''
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.metric = evaluate.load("sacrebleu")
        self.lr = lr
        self.total_steps = total_steps

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        dif = input_ids.shape[-1] - labels.shape[-1]
        if dif > 0:
            # FIX 1: Pad labels with -100 so loss ignores padding positions
            labels = torch.nn.functional.pad(labels, (0, dif), value=-100)
        elif dif < 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, -dif), value=self.tokenizer.pad_token_id)
            inputs['attention_mask'] = torch.nn.functional.pad(inputs['attention_mask'], (0, -dif), value=0)
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # FIX 2: Use self.device instead of hardcoded 'cuda'
        inputs = self.tokenizer(
            batch['source'],
            padding=True,
            truncation=True,
            max_length=256,
            text_target=batch['target'],
            return_tensors='pt'
        ).to(self.device)
        outputs = self.forward(inputs)
        loss = outputs.loss
        self.log_dict({'train_loss': loss}, batch_size=len(batch['source']))
        return loss

    def validation_step(self, batch, batch_idx):
        # FIX 3: Added truncation=True and max_length=256 to match training_step
        inputs = self.tokenizer(
            batch['source'],
            padding=True,
            truncation=True,
            max_length=256,
            text_target=batch['target'],
            return_tensors='pt'
        ).to(self.device)  # FIX 2: Use self.device instead of hardcoded 'cuda'

        loss = self.forward(inputs).loss

        # BLEU evaluation
        # outputs = self.model.generate(
        #     input_ids=inputs['input_ids'],
        #     attention_mask=inputs['attention_mask'],
        #     max_new_tokens=128
        # )
        # hyp = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # ref = batch['target']
        # bleu = self.metric.compute(predictions=hyp, references=ref)

        metrics = {'val_loss': loss}#, 'val_bleu': bleu['score']}
        self.log_dict(metrics, batch_size=len(batch['source']))
        return metrics

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        # FIX 4: Use self.total_steps instead of hardcoded 10000
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1, end_factor=1/3, total_iters=self.total_steps
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

class DecoderOnlyTranslationModel(TranslationModel):
    def _tokenize_source(self, sources):
        self.tokenizer.padding_side = 'left'   # enforce before every call
        return self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(self.device)

    def _tokenize_target(self, targets):
        self.tokenizer.padding_side = 'right'
        ids = self.tokenizer(
			targets,
			padding=True,
			truncation=True,
			max_length=256,
			return_tensors='pt'
		).input_ids.to(self.device)
        self.tokenizer.padding_side = 'left'
        ids[ids == self.tokenizer.pad_token_id] = -100
        return ids

    def training_step(self, batch, batch_idx):
        inputs = self._tokenize_source(batch['source'])
        inputs['labels'] = self._tokenize_target(batch['raw_target'])
        loss = self.forward(inputs).loss
        self.log_dict({'train_loss': loss}, batch_size=len(batch['source']))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self._tokenize_source(batch['source'])
        inputs['labels'] = self._tokenize_target(batch['raw_target'])
        loss = self.forward(inputs).loss

        # outputs = self.model.generate(
        #     input_ids=inputs['input_ids'],
        #     attention_mask=inputs['attention_mask'],
        #     max_new_tokens=128
        # )
        # # For decoder-only models, strip the input prompt tokens from the output
        # prompt_len = inputs['input_ids'].shape[-1]
        # outputs = outputs[:, prompt_len:]

        # hyp = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # ref = batch['target']
        # bleu = self.metric.compute(predictions=hyp, references=batch['raw_target'])
        metrics = {'val_loss': loss}#, 'val_bleu': bleu['score']}
        self.log_dict(metrics, batch_size=len(batch['source']))
        return metrics

def get_url(model_name):
    '''
    Returns the url of the model to download
    Parameters:
        model_name (str): Name of the model to download

    Returns:
        str: URL of the model
    '''
    urls = {
        'mbart':   'facebook/mbart-large-50-many-to-many-mmt',
        'm2m':     'facebook/m2m100_418M',
        'flant5':  'google/flan-t5-base',
        'nllb':    'facebook/nllb-200-distilled-600M',
        'llama':   'meta-llama/Llama-3.2-1B-Instruct',
        'qwen':    'Qwen/Qwen2.5-VL-7B-Instruct',
        'eurollm': 'utter-project/EuroLLM-1.7B',
        'gemma':   'google/gemma-3-4b-it',
    }
    if model_name not in urls:
        print('Model not implemented: {0}'.format(model_name))
        sys.exit(1)
    return urls[model_name]


def load_datasets(args):
    '''
    Loads the training and development datasets
    '''
    prompter = get_prompter(args.model_name, args.source, args.target)

    shards = [f"{args.folder}train.{args.source}",
              f"{args.folder}train.{args.target}"]
    training = MosesCorpus(shards[0], shards[1], prompter=prompter)

    shards = [f"{args.folder}dev.{args.source}",
              f"{args.folder}dev.{args.target}"]
    development = MosesCorpus(shards[0], shards[1], prompter=prompter)
    return training, development


def check_parameters(args):
    args.source_code = check_language_code(args.source) if args.model_name == 'mbart' else args.source
    args.target_code = check_language_code(args.target) if args.model_name == 'mbart' else args.target
    return args


def read_parameters():
    parser = argparse.ArgumentParser(description='Train a model for translation.')
    parser.add_argument("-src", "--source", required=True, help="Source Language")
    parser.add_argument("-trg", "--target", required=True, help="Target Language")
    parser.add_argument("-dir", "--folder", required=True, help="Folder where is the dataset")
    parser.add_argument('-model', '--model_name', default='mbart', choices=NAMES, help='Model to train')
    parser.add_argument('-lora', '--lora', action='store_true', help='Whether to use Low-Rank Adaptation or not')
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate of the optimizer')
    parser.add_argument('-log', '--log_dir', type=str, default='logs', help='Directory to save the logs')
    args = parser.parse_args()
    return args


def main():
    args = read_parameters()
    args = check_parameters(args)
    print(args)

    # FIX 5: Resolve device once; works on CPU, single GPU, and multi-GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metric = load("sacrebleu")
    model, tokenizer = load_model(get_url(args.model_name), args, device)

    if args.lora:
        model = apply_lora(model)

    # FIX 6: Cap num_workers to a safe ceiling to avoid CUDA dataloader deadlocks
    num_workers = min(int(os.cpu_count() * 0.75), 4)

    train_dataset, dev_dataset = load_datasets(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    # FIX 4: Compute total_steps from actual dataset size so LR schedule is meaningful
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.epochs

    # FIX 7: val_check_interval capped to dataset length so validation always runs
    val_check_interval = min(5000, steps_per_epoch)

    # FIX 8: Gradient accumulation condition fixed (<=32 instead of <32)
    accumulate = max(1, 32 // args.batch_size)

    fp16 = 't5' not in args.model_name
    
    DECODER_ONLY_MODELS = {'llama', 'qwen', 'eurollm', 'gemma'}
    model_class = DecoderOnlyTranslationModel if args.model_name in DECODER_ONLY_MODELS else TranslationModel
    translator = model_class(model, tokenizer, lr=args.learning_rate, total_steps=total_steps)

    callbacks = [
        L.pytorch.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2, min_delta=0.1),
        L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss', mode='min', save_top_k=3, save_weights_only=True,
            dirpath=f'models/{args.model_name}_{args.source+args.target}'
        )
    ]
    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=args.log_dir, name=f'{args.model_name}_{args.source+args.target}'
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        # precision='16-mixed' if fp16 else '32',
        logger=logger,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulate,
        default_root_dir=f'models/{args.model_name}_{args.source+args.target}',
        callbacks=callbacks
    )

    trainer.validate(model=translator, dataloaders=dev_dataloader)
    trainer.fit(model=translator, train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader)


if __name__ == '__main__':
    main()