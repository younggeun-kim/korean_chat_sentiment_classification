# BERT classification과 거의 완전 동일함
# 실험상 XLM_roberta는 데이터 개수가 적으면 정확도가 별로인 것 같다
Max_len = 56 # Max_len 길이 달라지면 정확도에 큰 영향을 미침
Batch_size = 16
num_category = 3
lr = 2e-5
device='cuda'
EPOCHS=4

def train_one_epoch(data_loader, model, optimizer, device, loss_fn):
  
  model.train()
  tk0 = tqdm(data_loader, total=len(data_loader))
  total_loss = 0.0
  
  for bi, d in enumerate(tk0):
      input_ids, token_type_ids, attn_mask_ids, label = d
      input_ids = input_ids.to(device, dtype=torch.long)
      token_type_ids = token_type_ids.to(device, dtype=torch.long)
      attn_mask_ids = attn_mask_ids.to(device, dtype=torch.long)
      label = label.to(device, dtype=torch.long)

      model.zero_grad()
      output = model(input_ids, token_type_ids, attn_mask_ids)
      #print(output.size())
      #print(label.size())
      loss = loss_fn(output, label.view(-1))
      
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if bi % 100 ==0:
          print(f"loss:{loss}")

  avg_train_loss = total_loss / len(data_loader) 
  print(" Average training loss: {0:.2f}".format(avg_train_loss))  

def eval_one_epoch(data_loader, model,  device, loss_fn):

  model.eval()
  tk0 = tqdm(data_loader, total=len(data_loader))
  fin_targets = []
  fin_outputs = []
  
  with torch.no_grad():

    for bi, d in enumerate(tk0):
      input_ids, token_type_ids, attn_mask_ids, label = d
      input_ids = input_ids.to(device, dtype=torch.long)
      token_type_ids = token_type_ids.to(device, dtype=torch.long)
      attn_mask_ids = attn_mask_ids.to(device, dtype=torch.long)
      label = label.to(device, dtype=torch.long)

      output = model(input_ids, token_type_ids, attn_mask_ids)
      loss = loss_fn(output, label.view(-1))

      output = output.detach().cpu().numpy()
      label = label.detach().cpu().numpy()
      pred = np.argmax(output, axis=1).flatten()

      fin_targets.extend(label.tolist())
      fin_outputs.extend(pred.tolist()) 

    
  return fin_outputs, fin_targets
  
def fit(train_dataloader, valid_dataloader, model, EPOCHS=3):
  loss_fn = nn.CrossEntropyLoss() #loss
  optimizer = torch.optim.AdamW(model.parameters(),lr=lr) #optimizer

  for i in range(EPOCHS):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    train_one_epoch(train_dataloader, model, optimizer, device, loss_fn)
    print('EVAL')
    outputs, targets = eval_one_epoch(valid_dataloader, model,  device, loss_fn)
    targets = np.array(targets)
    auc = accuracy_score(targets, outputs)
    print(f"auc;{auc}") 
    torch.save(model ,PATH+f'cls_{i+1}')
    
fit(train_dataloader, valid_dataloader, electra_cls)      
