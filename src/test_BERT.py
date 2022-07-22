from transformers import AutoTokenizer, AutoModel, AutoConfig, utils
utils.logging.set_verbosity_error()  # Suppress standard warnings

config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_hidden_layers = 14
config.memory_flag = False
config.output_attentions = True
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", config = config)

text = "Mrs Jackson loves dogs very much. She just adopted her third [MASK] yesterday. She might adopt her fourth [MASK] today."
inputs = tokenizer.encode(text, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

from bertviz import head_view, model_view
head_view(attention, tokens)
model_view(attention, tokens)