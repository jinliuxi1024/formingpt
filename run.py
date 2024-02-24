import sys
sys.path.append("..")
# sys.path.append("..")是为了导入上一级目录的模块
import torch
from mymingpt.model import GPTForLanguageModel
model = GPTForLanguageModel.from_pretrained('gpt2')
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
logits = model(input_ids)['logits']
from mymingpt.text_generator import GPTTextGernerator
text_generator = GPTTextGernerator.from_pretrained('gpt2')
print(text_generator.generate('My name is Clara and I am'))