
import torch
import pandas as pd
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split

path = "Participants_Data/"
train = pd.read_csv(os.path.join(path, "Train.csv"))
test = pd.read_csv(os.path.join(path, "Test.csv"))
PRE_TRAINED_MODEL_NAME = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


class SentiDataset(Dataset):

    def __init__(self, text, targets, tokenizer, max_len, infer):
        self.reviews = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.infer = infer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        text = self.reviews[item]
        input_ids, attention_mask = self._preprocess(text)
        if not self.infer:
            return input_ids.flatten(), attention_mask.flatten(), self.targets[item]
        else:
            return input_ids.flatten(), attention_mask.flatten()

    @staticmethod
    def _preprocess(text):
        out = tokenizer.encode_plus(text, padding="max_length", max_length=128, truncation=True,
                                    add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        return out['input_ids'], out['attention_mask']


def create_Dataloader(text, targets, tokenizer, max_length, batch_size, infer=False):
    data_set = SentiDataset(text, targets, tokenizer, max_length, infer)
    return DataLoader(data_set, batch_size=batch_size, num_workers=2)


text = train.Product_Description
targets = train.Sentiment

test_data = test.Product_Description

train_data, val_data = train_test_split(train, test_size=0.2, shuffle=True)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
train_text = train_data.Product_Description
train_targets = train_data.Sentiment
val_text = val_data.Product_Description
val_targets = val_data.Sentiment

test_data = test.Product_Description
train_loader = create_Dataloader(train_text, train_targets, tokenizer, 64, 128)
val_loader = create_Dataloader(val_text, val_targets, tokenizer, 64, 128)


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.linear(out[:, 0, :])


model = SentimentClassifier(
    4).cuda() if torch.cuda.is_available() else SentimentClassifier(4)
for name, param in model.named_parameters():
    if name.startswith('bert'):
        pass
        #param.requires_grad = False


print("Trainable parametes:", sum(p.numel()
                                  for p in model.parameters() if p.requires_grad))


train_length = len(train_loader)
val_length = len(val_loader)
history = defaultdict(list)


def training(train_loader, num_epoch, model, loss_fn, optimizer, train_length, val_loader, val_length):
    model.train()
    best_val_loss = 10
    for epoch in range(1, num_epoch+1):
        loss_train = 0.0
        correct_predictions = 0
        for input_ids, attention_mask, target in train_loader:
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                target = target.cuda()

            output = model(input_ids, attention_mask)
            l = loss_fn(output, target)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            print("Batch Loss:", l.item())
            loss_train += l.item()

        if epoch == 1 or epoch % 1 == 0:
            epoch_train_loss = loss_train/train_length
            history['train_loss'].append(epoch_train_loss)
            print('{} Epoch {}, Train Loss {}'.format(
                datetime.datetime.now(), epoch, epoch_train_loss))
        model = model.eval()

        loss_val = 0
        correct_predictions = 0

        with torch.no_grad():
            for input_ids, attention_mask, targets in val_loader:
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    targets = targets.cuda()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                loss_val += l.item()

            print(correct_predictions.double()/val_length, loss_val/val_length)

    return history


loss_fn = nn.CrossEntropyLoss().cuda(
) if torch.cuda.is_available() else nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([{'params': model.bert.parameters(), 'lr': 0.00001}, {
                              'params': model.linear.parameters(), 'lr': 0.1}])
training(train_loader, 5, model, loss_fn, optimizer,
         train_length, val_loader, val_length)

test_loader = create_Dataloader(text=test_data, targets=None,
                                tokenizer=tokenizer, max_length=64, batch_size=128, infer=True)

train_length = len(train_loader)


def validation(loader, num_epoch, model):
    model.eval()
    history = torch.zeros(1, 4)
    with torch.no_grad():
        for epoch in range(1, num_epoch+1):
            loss_train = 0.0
            correct_predictions = 0
            for input_ids, attention_mask in loader:
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()

                output = model(input_ids, attention_mask).cpu()
                history = torch.cat((history, output), dim=0)
                # Setting model to evaluation mode

    return history


ouput = validation(test_loader, 1, model)
out = torch.softmax(ouput, dim=1)
a = pd.DataFrame(out.numpy())
a.to_csv("submit.csv", index=False)
