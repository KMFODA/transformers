from importlib import reload
import sys, inspect

from tokenizers import Tokenizer
sys.path.append(".")

# %load_ext autoreload
# %autoreload 2
from transformers import BertModel, BertConfig, BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Mrs Jackson loves dogs very much.", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
