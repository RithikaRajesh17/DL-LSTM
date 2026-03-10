# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

This notebook tackles Named Entity Recognition (NER), an NLP task focused on identifying and classifying named entities within text. It employs a Bi-directional Long Short-Term Memory (Bi-LSTM) network in PyTorch to train a model. The objective is to accurately extract and categorize entities like person, location, and organization from input sentences.


## DESIGN STEPS

1. Load the NER dataset.

2. Prepare the dataset for training.

3. Create and train your model.

4. Predict the output for the test data and compare it with the actual value.





## PROGRAM

### Name:Rithika R

### Register Number:212224240136

```python
class BiLSTMTagger(nn.Module):
    def __init__(self,vocab_size,target_size,embedding_dim=50,hidden_dim=100):
      super(BiLSTMTagger, self).__init__()
      self.embedding=nn.Embedding(vocab_size,embedding_dim)
      self.dropout=nn.Dropout(0.1)
      self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True) 
      self.fc=nn.Linear(hidden_dim*2,target_size)



    def forward(self,x):
      x=self.embedding(x)
      x=self.dropout(x)
      x,_=self.lstm(x)
      return self.fc(x)


model =BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses,val_losses=[],[]
    for epoch in range(epochs):
      model.train()
      total_loss=0
      for batch in train_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss=0
      with torch.no_grad():
        for batch in test_loader:
          input_ids=batch["input_ids"].to(device)
          labels=batch["labels"].to(device)
          outputs=model(input_ids)
          loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
          val_loss+=loss.item()
      val_losses.append(val_loss)
      print(f"Epoch (epoch+1): Train Loss={total_loss:.4f},Val Loss={val_loss:.4f}")    

    return train_losses, val_losses
```

### OUTPUT

## Loss Vs Epoch Plot

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/009dadb2-d064-4bcf-9fad-d3df89b8900a" />

### Sample Text Prediction
<img width="799" height="552" alt="image" src="https://github.com/user-attachments/assets/dfe08e15-bcaa-4c71-825a-cfe56da73e83" />

## RESULT
the program has been done successfully.
