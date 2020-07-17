from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Custom_dataset(Dataset):
  def __init__(self, input_ids, token_type_ids, attn_mask_ids, label):
    self.input = input_ids
    self.token_type = token_type_ids
    self.attn_mask = attn_mask_ids
    self.label = label
 
  def __len__(self):
    return len(self.input)  

  def __getitem__(self, idx):
    input = self.input[idx]
    tok = self.token_type[idx]
    attn = self.attn_mask[idx]
    label = torch.tensor(self.label[idx], dtype= torch.long).view(-1)

    return input, attn, tok ,label
    
train_dataset = Custom_dataset(input_ids_t, token_type_ids_t, mask_ids_t, train_data['label'])
valid_dataset = Custom_dataset(input_ids_v, token_type_ids_v, mask_ids_v, valid_data['label'])
train_dataloader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=True)
