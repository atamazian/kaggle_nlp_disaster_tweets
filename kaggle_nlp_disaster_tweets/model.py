import numpy as np
import pandas as pd
import os
from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics import F1
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

class LitDataNLP(LightningDataModule):
    def __init__(self,
        train_df,
        valid_df,
        test_df,
        model_name='roberta-base',
        max_length: int = 64,
        batch_size: int = 32
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name.
                         model_max_length=max_length)
        self.max_length = max_length
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size

    def extract_data(self, df, labeled=True):            
        df = df[df['text']!='']
        texts = df.text.values
        indices = self.tokenizer.batch_encode_plus(texts,
                    max_length=self.max_length, add_special_tokens=True, 
                    return_attention_mask=True, pad_to_max_length=True,
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
        
        self.model = AutoModelForSequenceClassification(model_name, num_labels=2)
        self.f1_score = F1(num_classes=2)
        
    def forward(self, b_input_ids, b_input_mask, b_labels):
        output = self.base_model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels)
        return output
    
    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask, b_labels)
        loss = z[0]
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask, b_labels)
        val_loss = z[0]
        logits = z[1]
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_f1_score', self.f1_score(logits, b_labels), prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.base_model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=self.warmup, 
                                            num_training_steps=189*self.epochs)
        return [optimizer], [scheduler]
    
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
