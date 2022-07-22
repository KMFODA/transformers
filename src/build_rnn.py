from importlib import reload
from re import L
import sys, inspect

from tokenizers import Tokenizer
import torch
import numpy
sys.path.insert(0, "/Users/karimfoda/Documents/STUDIES/PYTHON/BERTVIZ/")

import os
os.environ['PYTHONBREAKPOINT'] = "0"

from transformers import BertModel, AutoConfig, BertTokenizer, BertForMaskedLM, utils
from bertviz import head_view, model_view, util

def pad(tensor, max_sentence_length = 32):
    padding_length = max_sentence_length - tensor.size()[0]
    new_tensor = torch.cat((tensor,torch.tensor([0]*padding_length)), dim=0)
    return new_tensor

config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_hidden_layers = 12
config.memory_flag = True
config.output_attentions = True

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased", config = config)

# Model Info
# for name, param in model.named_parameters():                
#     print(name, param.size())

# print(model.bert.encoder.layer[12].memory)
# print(model.bert.encoder.layer[12].output_attention.self.value.weight[0])
# print(model.bert.encoder.layer[12].output_attention.self.query.weight[0])
# print(model.bert.encoder.layer[12].output_attention.self.key.weight[0])

# text = "Mrs Jackson loves dogs very much. She just adopted her third [MASK] yesterday."
text = [
    "Mrs Jackson loves dogs very much. She just adopted her third [MASK] yesterday. She might adopt her fourth [MASK] today.",
    "Mrs Belmond doesn't like dogs very much. She will never adopt a [MASK]."
]

# TODO Train using BS = 1
# TODO Tweak HF Trainer, edit loss function

if config.memory_flag == True:
    
    if type(text) != str:
        
        inputs = tokenizer(text, padding = "max_length", return_tensors="pt")
        
        # 1. Sentence Label
        sentence_label = torch.tensor([[0 if ((tokenizer.decode([i]) != ".") and (tokenizer.decode([i]) != "?")) else 1 for i in j] for j in inputs["input_ids"]])
        doc_sentence_label_id = []
        for document in sentence_label:
            sentence_label_id = []
            sentence_id = 0
            for token_sentence_id in document:
                if token_sentence_id == 0:
                    sentence_label_id.append(sentence_id)
                    continue
                elif token_sentence_id == 1:
                    sentence_label_id.append(sentence_id)
                    sentence_id += 1
            doc_sentence_label_id.append(sentence_label_id)

        sentence_flag_ids = torch.tensor(doc_sentence_label_id)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        # TODO Add to Config
        max_sentence_length = 32
        max_sentence_count = int(max([e.max() for e in sentence_flag_ids]))
        batch_size = input_ids.size()[0]
        new_inputs_ids =  torch.zeros(size=(batch_size, max_sentence_count,max_sentence_length))
        new_token_type_ids =  torch.zeros(size=(batch_size, max_sentence_count,max_sentence_length))
        new_attention_mask =  torch.zeros(size=(batch_size, max_sentence_count,max_sentence_length))
        for i in range(0, len(input_ids)):
            sentence_count = sentence_flag_ids[i].max()
            for sentence_no in range(0, sentence_count):
                new_inputs_ids[i][sentence_no] = pad(input_ids[i][sentence_flag_ids[i] == sentence_no])
                new_token_type_ids[i][sentence_no] = pad(token_type_ids[i][sentence_flag_ids[i] == sentence_no])
                new_attention_mask[i][sentence_no] = pad(attention_mask[i][sentence_flag_ids[i] == sentence_no])

        input_ids = new_inputs_ids.type(torch.long)
        token_type_ids = new_token_type_ids.type(torch.long)
        attention_mask = new_attention_mask.type(torch.long)

        output = model(**inputs, sentence_flag_ids = sentence_flag_ids)
        # breakpoint()
        token_logits = output.logits
        attentions = output.attentions

        for batch in range(0, input_ids.size()[0]):
            for sentences in range(0, input_ids.size()[1]):
                if len(input_ids.size())==2:
                    # TODO
                    continue
                for tokens in range(0, input_ids.size()[2]):

                    if int(input_ids[batch][sentences][tokens]) == int(tokenizer.mask_token_id):
                        # breakpoint()
                        print(input_ids[batch][sentences][tokens])
                        mask_token_logits=token_logits[batch, sentences, tokens, :]
                        top_5_tokens = torch.topk(mask_token_logits, 5, dim=0).indices.tolist()    
                        print(tokenizer.decode(top_5_tokens))

    else:
        inputs = tokenizer([i.strip() + "." for i in text.split(".") if i != ""], padding = "longest", return_tensors="pt")
        outputs = model(**inputs)
        token_logits = outputs.logits
        attentions = outputs.attentions

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        for sentences in range(0, input_ids.size()[0]):
            if int(sum(input_ids[sentences] == tokenizer.mask_token_id)) != 0:
                for tokens in range(0, len(input_ids[sentences])):
                    if int(input_ids[sentences][tokens]) == int(tokenizer.mask_token_id):
                        # breakpoint()
                        print(input_ids[sentences][tokens])
                        mask_token_logits=token_logits[sentences, tokens, :]
                        top_5_tokens = torch.topk(mask_token_logits, 5, dim=0).indices.tolist()    
                        print(tokenizer.decode(top_5_tokens))

        # tokens = tokenizer.convert_ids_to_tokens(input_ids[1]) 
        # model_view(attentions, tokens)

        for sentence_attention in range(0, input_ids.size()[0]):
            new_attention_layer = ()
            memory_layer = ()
            for i, attention_layer in enumerate(attentions):
                if i == config.num_hidden_layers:
                    memory_layer = memory_layer + (attention_layer[sentence_attention].view(((1,)+tuple(attention_layer[sentence_attention].size()))),)
                    break
                new_attention_layer = new_attention_layer + (attention_layer[sentence_attention].view(((1,)+tuple(attention_layer[sentence_attention].size()))),)
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids[sentence_attention]) 
            model_view( new_attention_layer, tokens)

            tokens = tokenizer.convert_ids_to_tokens(input_ids[sentence_attention]) 
            model_view( memory_layer, tokens)


        # mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        # mask_token_logits = token_logits[0, mask_token_index, :]
        # # Pick the [MASK] candidates with the highest logits
        # top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        # print(tokenizer.decode(top_5_tokens))

else:
    # BERTVIZ with no RNN
    config.memory_flag = False
    model_2 = BertForMaskedLM.from_pretrained("bert-base-uncased", config = config)

    # inputs_2 = tokenizer([text,text], padding = 'longest', return_tensors='pt')
    inputs_2 = tokenizer([text], padding = 'longest', return_tensors='pt')
    inputs_2 = tokenizer(text, padding = 'longest', return_tensors='pt')

    outputs_2 = model_2(**inputs_2)
    token_logits_2 = outputs_2.logits
    attentions_2 = outputs_2.attentions
    attentions_2 = outputs_2[-1]  # Output includes attention weights when output_attentions=True
    
    input_ids_2 = inputs_2["input_ids"]
    token_type_ids_2 = inputs_2["token_type_ids"]
    attention_mask_2 = inputs_2["attention_mask"]

    for sentences in range(0, input_ids_2.size()[0]):
        if int(sum(input_ids_2[sentences] == tokenizer.mask_token_id)) != 0:
            for tokens in range(0, len(input_ids_2[sentences])):
                if int(input_ids_2[sentences][tokens]) == int(tokenizer.mask_token_id):
                    breakpoint()
                    print(input_ids_2[sentences][tokens])
                    mask_token_logits=token_logits_2[sentences, tokens, :]
                    top_5_tokens = torch.topk(mask_token_logits, 5, dim=0).indices.tolist()    
                    print(tokenizer.decode(top_5_tokens))

    for sentence_attention in range(0, input_ids_2.size()[0]):
        # break
        # sentence_attention = 1
        new_attention_layer_2 = ()
        for attention_layer in attentions_2:
            new_attention_layer_2 = new_attention_layer_2 + (attention_layer[sentence_attention].view(((1,)+tuple(attention_layer[sentence_attention].size()))),)
        tokens_2 = tokenizer.convert_ids_to_tokens(input_ids_2[sentence_attention]) 
        model_view( new_attention_layer_2, tokens_2)

        # attentions_all_layers_2 = [attention_layer[sentence_attention] for attention_layer in attentions_2])
        # attentiona_layers_stacked_2 = torch.stack(attentions_all_layers_2)[0]
        # model_view(attentiona_layers_stacked_2.view(((1,)+tuple(attentiona_layers_stacked_2.size()))), tokens_2)

############################## ARCHIVE ##############################
# # Find the location of [MASK] and extract its logits
# mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
# mask_token_logits = token_logits[0, mask_token_index, :]
# # Pick the [MASK] candidates with the highest logits
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# for token in top_5_tokens:
#     # breakpoint()
#     print(f"'>>> {tokenizer.decode([token])}'")

# outputs = model(**inputs)
# # breakpoint()
# print(outputs)


# from bertviz import model_view
# utils.logging.set_verbosity_error() 
# model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)  # Configure model to return attention values
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# inputs = tokenizer.encode(text, return_tensors='pt')  # Tokenize input text
# outputs = model(inputs)  # Run model
# attention = outputs[-1]  # Retrieve attention from model outputs
# tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 
# model_view(attention[12], tokens)  # Display model view

# import numpy
# sentence_end_ids  = torch.tensor([numpy.where(e == 1)[0]+1 for e in sentence_label])
# sentence_start_ids = torch.tensor([[0 if z == 0 else e[z-1] for z,_ in enumerate(e)] for e in sentence_end_ids])

# i = 0
# new_input_ids = [[]*len(inputs['input_ids'])]
# for e,z in zip(sentence_start_ids, sentence_end_ids):
#     break
#     for e_2, z_2 in zip(e,z):
#         break
#         new_input_ids[i] = new_input_ids[i].append(inputs['input_ids'][0][z_2:e_2])
#         new_input
    
# breakpoint()
# inputs = tokenizer("Mrs Jackson loves dogs very much. She just adopted her third one. Test this again.", padding = "max_length", return_tensors="pt")
