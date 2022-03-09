import numpy as np
import pandas as pd
import os
import re
from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics import F1
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, \
                         get_linear_schedule_with_warmup

def preprocess(text):
    text=text.lower()
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http?:\/\/.*[\r\n]*', '', text)
    #Replace &amp, &lt, &gt with &,<,> respectively
    text=text.replace(r'&amp;?',r'and')
    text=text.replace(r'&lt;',r'<')
    text=text.replace(r'&gt;',r'>')
    #remove mentions
    text = re.sub(r"(?:\@)\w+", '', text)
    #remove non ascii chars
    text=text.encode("ascii",errors="ignore").decode()
    #remove some puncts (except . ! ?)
    text=re.sub(r'[:"#$%&\*+,-/:;<=>@\\^_`{|}~]+','',text)
    text=re.sub(r'[!]+','!',text)
    text=re.sub(r'[?]+','?',text)
    text=re.sub(r'[.]+','.',text)
    text=re.sub(r"'","",text)
    text=re.sub(r"\(","",text)
    text=re.sub(r"\)","",text)
    
    text=" ".join(text.split())
    return text

class LitDataNLP(LightningDataModule):
    def __init__(self,
        train_df,
        valid_df,
        test_df,
        model_name = 'roberta-base',
        max_length = 64,
        batch_size = 32,
        clean_text = False
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_length)
        self.max_length = max_length
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.clean_text = clean_text

    def extract_data(self, df, labeled=True):            
        df = df[df['text'] != '']
        if self.clean_text:
            df['text'] = df['text'].apply(preprocess)
        texts = df.text.values.tolist()
        indices = self.tokenizer.batch_encode_plus(texts,
                    max_length=self.max_length, add_special_tokens=True, 
                    return_attention_mask=True, padding='max_length',
                    truncation=True)
        input_ids=torch.tensor(np.array(indices["input_ids"]))
        attention_masks=torch.tensor(np.array(indices["attention_mask"]), dtype=torch.long)
        
        if labeled:
            labels = torch.tensor(df.target.values, dtype=torch.long)
            return input_ids, labels, attention_masks
        else:
            return input_ids, attention_masks
       
    def setup(self, stage=None):
        self.train_inputs, self.train_labels, self.train_masks = self.extract_data(df=self.train_df)
        self.valid_inputs, self.valid_labels, self.valid_masks = self.extract_data(df=self.valid_df)
        self.test_inputs, self.test_masks = self.extract_data(df=self.test_df, labeled=False)
        
    def train_dataloader(self) -> DataLoader:
        train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
        train_sampler = RandomSampler(train_data)
        return DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
    
    def val_dataloader(self) -> DataLoader:
        validation_data = TensorDataset(self.valid_inputs, self.valid_masks, self.valid_labels)
        validation_sampler = SequentialSampler(validation_data)
        return DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)
        
    def test_dataloader(self) -> DataLoader:
        test_data = TensorDataset(self.test_inputs, self.test_masks)
        test_sampler = SequentialSampler(test_data)
        return DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)

class LitNLPModel(LightningModule):
    def __init__(self, 
                 model_name,
                 epochs,
                 
                 lr: float = 6e-6,
                 warmup: int = 0):
      
        super().__init__()
        
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup
        
        config = AutoConfig.from_pretrained(model_name, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        self.f1_score = F1(num_classes=2)
        
    def forward(self, b_input_ids, b_input_mask, b_labels=None):
        output = self.model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels)
        return output
    
    def training_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels = batch
        z = self(b_input_ids, b_input_mask, b_labels)
        loss, logits = z[0], z[1]
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids, b_input_mask, b_labels = batch
        z = self(b_input_ids, b_input_mask, b_labels)
        val_loss, logits = z[0], z[1]
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_f1_score', self.f1_score(logits, b_labels), prog_bar=True)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
