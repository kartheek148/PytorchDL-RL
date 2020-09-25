
import torch.nn as nn
from data import create_Dataloader
import os
import pandas as pd
import torch
from transformers import RobertaModel, get_linear_schedule_with_warmup
import datetime
import torch.nn.functional as F
from collections import defaultdict
from sklearn.model_selection import train_test_split

path = "Lnmwdnq3YcF7F3YsJncp/data"


os.chdir(path)
train = pd.read_json(os.path.join(path, "train.jsonl"), lines=True)
test = pd.read_json(os.path.join(path, "test.jsonl"), lines=True)
dev = pd.read_json(os.path.join(path, "dev.jsonl"), lines=True)

data = pd.concat([train, dev], axis=0, ignore_index=True)
train_data, val_data = train_test_split(data, test_size=0.2, shuffle=True)
train_data = train_data.reset_index(drop=True)[:100]
val_data = val_data.reset_index(drop=True)[:100]


train_loader = create_Dataloader(train_data, 2)
val_loader = create_Dataloader(val_data, 128)


torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

# load pretrained models, using ResNeSt-50 as an example
RESNEST = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
RESNEST.fc = nn.Identity()

ROBERTA = RobertaModel.from_pretrained("roberta-base")


class DecoderLayer(nn.Module):
    def __init__(self, edim, nhead, dim_feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            edim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(edim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, edim)
        self.norm1 = nn.LayerNorm(edim)
        self.norm2 = nn.LayerNorm(edim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mem, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        query = src
        key = mem
        value = mem
        print(src.shape, mem.shape)
        src2 = self.multihead_attn(
            query, key, value, key_padding_mask=tgt_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class HMModel(nn.Module):
    def __init__(self, edim):
        super(HMModel, self).__init__()
        self.resnest = RESNEST
        self.roberta = ROBERTA
        self.decoderLayer = DecoderLayer(edim, 8, 1024, 0.2)
        self.linear_bert = nn.Linear(768, edim)
        #self.norm = nn.Sequential(nn.Linear(edim,2))
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoderLayer, num_layers=4)

    def forward(self, image, input_ids, attention_mask):
        resnet_out = self.resnest(image)
        bert_out = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)[0]
        bert_out = self.linear_bert(bert_out).transpose_(0, 1)
        #attention_mask = attention_mask.transpose(0,1)
        #bert_out = bert_out.unsqueeze(0)
        resnet_out = resnet_out.view(4, -1, 512)
        print(bert_out.shape, resnet_out.shape, attention_mask.shape)
        out = self.decoder(tgt=resnet_out, memory=bert_out,
                           tgt_key_padding_mask=attention_mask)
        return out


model = HMModel(512)
loss_fn = nn.BCEWithLogitsLoss().cuda(
) if torch.cuda.is_available() else nn.BCEWithLogitsLoss()

for name, param in model.named_parameters():
    if name.startswith('roberta') or name.startswith("resnest"):
        param.requires_grad = False


print("Trainable parametes:", sum(p.numel()
                                  for p in model.parameters() if p.requires_grad))


optimizer = torch.optim.AdamW(model.parameters())


train_length = len(train_loader)
val_length = len(val_loader)
history = defaultdict(list)


def training(train_loader, num_epoch, model, loss_fn, optimizer, train_length, val_length):
    model.train()
    best_val_loss = 10
    for epoch in range(1, num_epoch+1):
        model.train()
        loss_train = 0.0
        correct_predictions = 0
        for image, input_ids, attention_mask, target in train_loader:
            if torch.cuda.is_available():
                image = image.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                target = target.cuda()

            output = model(image, input_ids, attention_mask)
            preds = torch.sigmoid(output)
            if torch.cuda.is_available():
                plabel = torch.where(preds > 0.5, torch.Tensor(
                    [1.0]).cuda(), torch.Tensor([0.0]).cuda())
            else:
                plabel = torch.where(preds > 0.5, torch.Tensor(
                    [1.0]), torch.Tensor([0.0]))

            correct_predictions += torch.sum(plabel == target)
            l = loss_fn(output, target)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            loss_train += l.item()

        if epoch == 1 or epoch % 1 == 0:
            epoch_train_loss = loss_train/train_length
            epoch_train_acc = correct_predictions.double()/train_length
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            print('{} Epoch {}, Train Loss {}, Train Acc {}'.format(
                datetime.datetime.now(), epoch, epoch_train_loss, epoch_train_acc))

        model.eval()        # Setting model to evaluation mode
        loss_train = 0.0
        correct_predictions = 0
        with torch.no_grad():
            for image, input_ids, attention_mask, target in val_loader:
                target = target.view(-1, 1)
                target = target.float()
                if torch.cuda.is_available():
                    image = image.cuda()
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    target = target.cuda()
                output = model(image, input_ids, attention_mask)
                preds = torch.sigmoid(output)
                if torch.cuda.is_available():
                    plabel = torch.where(preds > 0.5, torch.Tensor(
                        [1.0]).cuda(), torch.Tensor([0.0]).cuda())
                else:
                    plabel = torch.where(preds > 0.5, torch.Tensor(
                        [1.0]), torch.Tensor([0.0]))

                correct_predictions += torch.sum(plabel == target)
                l = loss_fn(output, target)
                loss_train += l.item()
            epoch_val_loss = loss_train/val_length
            epoch_val_acc = correct_predictions.double()/val_length
            history['train_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_val_acc)
            print('{}, Val Loss {}, Val Acc {}'.format(
                datetime.datetime.now(), epoch_val_loss, epoch_val_acc))
            if epoch_val_loss < best_val_loss:
                torch.save(model.state_dict(), 'best_model.pth')
                best_val_loss = epoch_val_loss
    return history
