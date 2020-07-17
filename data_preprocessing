PATH = '/content/gdrive/My Drive/'
from transformers import ElectraModel, ElectraTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaModel
electra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# data from https://github.com/songys/Chatbot_data
data = pd.read_csv(PATH+'ChatbotData .csv')

# train data valid data split
def valid_train_data(data, ratio=0.8): #ratio should under 1
  num = int(len(data)*ratio)
  data = data.sample(frac = 1)
  train_data = data[:num].reset_index()
  valid_data = data[num:].reset_index()

  return train_data, valid_data
  
# model에 들어갈 input을 만들어주기  
def text_processing(data, Max_len=36, fn = electra_tokenizer):
  input_ids = []
  mask_ids = []
  token_type_ids = []
  tokenizer = fn
  for i in range(len(data)):
    text1 = data['Q'][i]
    text2 = data['A'][i]
    # [cls], [sep] 추가 Max_len 길이 맞추기  
    encoded_line1 = tokenizer.encode(text1, add_special_tokens = True, max_length=Max_len, truncation=True) 
    encoded_line2 = tokenizer.encode(text2, add_special_tokens = True, max_length=Max_len, truncation=True) 
    pad_len = Max_len - len(encoded_line1) - len(encoded_line2)
    inp_len = len(encoded_line1) + len(encoded_line2)
    if fn == electra_tokenizer:
      pad_token = 0
    else: pad_token = 1  # if fn == xlm_tokenizer
    input = encoded_line1 + encoded_line2 + [pad_token]*pad_len #Padding
    token_type =  [0]*len(encoded_line1) + [1]*(Max_len-len(encoded_line1))
    mask = [1]*inp_len + [0]*pad_len    
    input = input[:Max_len]
    token_type = token_type[:Max_len]
    mask = mask[:Max_len]
    input_ids.append(torch.tensor(input))
    token_type_ids.append(torch.tensor(token_type))
    mask_ids.append(torch.tensor(mask))

  return input_ids, token_type_ids, mask_ids
  
train_data, valid_data = valid_train_data(data)
input_ids_t, token_type_ids_t, mask_ids_t = text_processing(train_data)
input_ids_v, token_type_ids_v, mask_ids_v = text_processing(valid_data)

#input_ids_t, token_type_ids_t, mask_ids_t = text_processing(train_data, fn = xlm_tokenizer)
#input_ids_v, token_type_ids_v, mask_ids_v = text_processing(valid_data, fn = xlm_tokenizer)
