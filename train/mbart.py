# Load model directly
from argparse import ArgumentParser

import evaluate
import torch
import transformers as trans

parser = ArgumentParser('Script que automatiza el fine-tunning de un modelo MBART.')
parser.add_argument('-args','--argument_file',help='Archivo yaml del que se toman los argumentos del entrenamiento.')
parser.add_argument('-src_tr','--source_train',default='EuTrans/train/training.es',help='Archivo con el conjunto de entrenamiento en lengua origen.')
parser.add_argument('-src_ev','--source_evaluation',default='EuTrans/dev/development.es',help='Archivo con el conjunto de evaluacion en lengua origen.')
parser.add_argument('-tgt_tr','--target_train',default='EuTrans/train/training.en',help='Archivo con el conjunto de entrenamiento en lengua destino.')
parser.add_argument('-tgt_ev','--target_evaluation',default='EuTrans/dev/development.en',help='Archivo con el conjunto de evaluacion en lengua destino.')
parser.add_argument('-out_dir','--output_dir',default='mbart',help='Directorio en que se escriben las predicciones del modelo y checkpoints.')
parser.add_argument('-a_out_dir','--overwrite_output_dir',action='store_true',default=False,help='Indica si se sobreescribe el directorio de salida.')
parser.add_argument('-bs','--batch_size',default=32,type=int,help='Tamaño de batch.')
parser.add_argument('-eval','--eval_strategy',default='epoch',choices=['no','epoch','steps'],help='Estrategia de evaluacion: al final de cada epoch o tras el numero de pasos especificado.')
parser.add_argument('-eval_steps','--eval_steps',default=100,type=int,help='Numero de actualizaciones antes de la siguiente evaluacion si la estrategia de evaluacion es steps.')
parser.add_argument('-lr','--learning_rate',default=1e-3,type=float,help='Tasa de aprendizaje.')
parser.add_argument('-opt','--optim',default='adamw_torch',choices=['adamw_hf','adamw_torch','adamw_torch_fused','adamw_apex_fused','adamw_anyprecision',
                                                                    'adafactor'], help='Optimizado a utilizar.')
parser.add_argument('-weight_decay','--weight_decay',default=0,type=float,help='Weight decay del optimizador.')
parser.add_argument('-e','--epochs',default=3,type=int,help='Numero de epocas que realiza el fine-tunning.')
parser.add_argument('-max_steps','--max_steps',type=int,help='Numero maximo de pasos que realiza el fine-tunning. Sobreescribe el numero de epocas.')
parser.add_argument('-save','--save_strategy',default='epoch',choices=['no','epoch','steps'],help='Estrategia de guardado de checkpoints: al final de cada epoch o tras el numero de pasos especificado.')
parser.add_argument('-save_steps','--save_steps',default=500,type=int,help='Numero de pasos a los que hacer el guardado.')
parser.add_argument('-save_limit','--save_total_limit',type=int,default=50,help='Numero de checkpoints a guardar. Si se supera se eliminan los mas antiguos.')
parser.add_argument('-model_only','--save_only_model',action='store_true',default=False,help='Indica si guardar solo el modelo en los checkpoints.')
parser.add_argument('-cpu','--use_cpu',action='store_true',default=False,help='Indica si usar la CPU en vez de CUDA.')
parser.add_argument('-workers','--num_workers',default=0,type=int,help='Número de procesos de CPU para el dataloader.')
parser.add_argument('-warm_start','--resume_from_checkpoint',action='store_true',default=False,help='Indica si empezar desde el modelo guardado en los checkpoints.')
args = parser.parse_args()

if args.max_steps:
    training_args = trans.Seq2SeqTrainingArguments(output_dir=args.output_dir,overwrite_output_dir=args.overwrite_output_dir,
                                            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                                            evaluation_strategy=args.eval_strategy,learning_rate=args.learning_rate,
                                            weight_decay=args.weight_decay,max_steps=args.max_steps,
                                            lr_scheduler_type='linear',
                                            save_strategy=args.save_strategy,save_steps=args.save_steps,save_total_limit=args.save_total_limit,
                                            save_only_model=args.save_only_model, use_cpu=args.use_cpu,eval_steps=args.eval_steps,
                                            dataloader_num_workers=args.num_workers, load_best_model_at_end=True,
                                            metric_for_best_model='loss',greater_is_better=False,
                                            optim=args.optim,
                                            resume_from_checkpoint=args.resume_from_checkpoint)
else:
    training_args = trans.Seq2SeqTrainingArguments(output_dir=args.output_dir,overwrite_output_dir=args.overwrite_output_dir,
                                            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                                            evaluation_strategy=args.eval_strategy,learning_rate=args.learning_rate,
                                            weight_decay=args.weight_decay,num_train_epochs=args.epochs,
                                            lr_scheduler_type='linear',
                                            save_strategy=args.save_strategy,save_steps=args.save_steps,save_total_limit=args.save_total_limit,
                                            save_only_model=args.save_only_model, use_cpu=args.use_cpu,eval_steps=args.eval_steps,
                                            dataloader_num_workers=args.num_workers, load_best_model_at_end=True,
                                            metric_for_best_model='loss',greater_is_better=False,
                                            optim=args.optim,
                                            resume_from_checkpoint=args.resume_from_checkpoint)

tokenizer = trans.AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
model = trans.AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

data_collator = trans.DataCollatorForSeq2Seq(tokenizer, model=model)

class Eutrans(torch.utils.data.Dataset):
    def __init__(self, source_file, target_file):
        self.src_lang = source_file.split('.')[-1]
        self.tgt_lang = target_file.split('.')[-1]
        with open(source_file,'r') as f:
            self.source = [line for line in f]
        with open(target_file,'r') as f:
            self.target = [line for line in f]
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        input = tokenizer(self.source[idx],text_target=self.target[idx])
        return input

train_data = Eutrans(args.source_train,args.target_train)
eval_data = Eutrans(args.source_evaluation,args.target_evaluation)

metric = evaluate.load('sacrebleu')


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

trainer = trans.Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)