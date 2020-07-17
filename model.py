import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer

electramodel = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")
num_category = 3

class ELECTRA_model(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
    self.dropout = nn.Dropout(0.3)
    self.gelu = nn.GELU()
    self.linear = nn.Linear(768, 768)
    self.out = nn.Linear(768,num_category)
 
  def forward(self, input_ids, token_type_ids, attn_mask_ids):
    #out_size = [batch_size, len, hidden_state]
    output = self.model(input_ids, attention_mask=attn_mask_ids, token_type_ids=token_type_ids)
    out = output[0]
    x = self.dropout(out[:,0,:])  # extract [CLS] token
    x = self.linear(x)
    x = self.gelu(x)
    x = self.dropout(x)
    logits = self.out(x)
 
    return logits

class XLM_ROBERTa_model(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
    self.dropout = nn.Dropout(0.3)
    self.gelu = nn.GELU()
    self.linear = nn.Linear(768, 768)
    self.out = nn.Linear(768,num_category)
 
  def forward(self, input_ids, token_type_ids, attn_mask_ids):

    output = self.model(input_ids, attention_mask=attn_mask_ids) # because XLM_RoBERTa doesn't have token_type_ids
    out = output[0]
    x = self.dropout(out[:,0,:])
    x = self.linear(x)
    x = self.gelu(x)
    x = self.dropout(x)
    logits = self.out(x)
 
    return logits
electra_cls = ELECTRA_model(electramodel).to(device)
