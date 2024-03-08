from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

beam = 4

# Cargar el modelo y el tokenizador
model_name = "Helsinki-NLP/opus-mt-en-es"  # modelo de traducción de inglés a español
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

# Texto de entrada en inglés
input_text = "¿Como puedo hacer inferencia con un modelo de la libreria transformers de Hugging Face?"

# Tokenizar el texto de entrada
#inputs = tokenizer(input_text, return_tensors="pt",output_scores=True)

# Obtener la salida del modelo
with torch.no_grad():
    paths = [tokenizer.eos_token]
    scores = torch.ones(beam)
    sm = torch.softmax(tokenizer.vocab_size)
    # POR CADA PASO:
    out_scores = torch.empty(beam*tokenizer.vocab_size)
    for i in len(paths):
        inputs = tokenizer(input_text, text_pair=paths[i], return_tensors="pt")
        outputs = model(**inputs)
        out_scores[i*tokenizer.vocab_size:(i+1)*tokenizer.vocab_size] = scores[i] * sm(outputs.logits[:,-1,:])
    scores, idx = torch.topk(out_scores, beam)
    paths = [paths[out_scores.size[0]//beam] + [idx[i] % out_scores.size[0]] for i in range(beam)]

# Obtener los logits de salida
logits = outputs.logits

# Obtener la secuencia de tokens más probable
predicted_ids = torch.argmax(logits, dim=-1)

# Decodificar la secuencia de tokens en texto
translated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print("Texto original (en inglés):", input_text)
print("Texto traducido (en español):", translated_text)
