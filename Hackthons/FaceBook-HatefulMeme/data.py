# %%
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image as io
from transformers import RobertaTokenizer
import torchvision.transforms.functional as F
import numpy as np


# %%
PRE_TRAINED_MODEL_NAME = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#%%
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

# %%
class HMMDataset(Dataset):
    def __init__(self, data, infer):
        self.image = data['img']
        self.text = data['text']
        self.infer = infer
        if not self.infer:
            self.label = data['label']
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __getitem__(self, item):
        image = io.open(self.image[item]).convert('RGB')
        image = self.transform(image)
        text = self.text[item]
        input_ids, attention_mask = self._preprocess(text)
        if not self.infer:
            return image,input_ids.flatten(),attention_mask.flatten(),self.label[item]
        else:
            return image,input_ids.flatten(),attention_mask.flatten()
    def __len__(self):
        return len(self.text)
        
    @staticmethod
    def _preprocess(text):
       out = tokenizer.encode_plus(text, padding = "max_length", max_length = 128,truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
       return out['input_ids'], out['attention_mask']
        


# %%
def create_Dataloader(data,batch_size,infer=False):
    data_set = HMMDataset(data,infer)
    return DataLoader(data_set, batch_size = batch_size,num_workers=0)



# %%
