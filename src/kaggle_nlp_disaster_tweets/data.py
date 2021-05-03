import numpy as np
import pandas as pd
import os
from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_lightning import LightningDataModule

class LitDataNLP(LightningDataModule):
    def __init__(self, 
        tokenizer,
        max_length: int = 64,
        train_df: Union[str, pd.DataFrame] = 'train.csv',
        valid_df: Union[str, pd.DataFrame] = 'valid.csv',
        batch_size: int = 32
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size

    def prepare_data(self, df) -> tuple:
        if isinstance(df, pd.DataFrame):
            pass
        elif isinstance(df, str):
            assert os.path.isfile(df), f'file not found: {df}'
            df = pd.read_csv(df)
        else:
            raise ValueError(f'unable to read Pandas DataFrame/CSV file: {df}')
            
        df = df[df['text']!='']
        df = df[['text', 'target']]
        texts = df.text.values
        labels = torch.tensor(df.target.values, dtype=torch.long)
        indices = self.tokenizer.batch_encode_plus(texts,
                    max_length=max_length, add_special_tokens=True, 
                    return_attention_mask=True, pad_to_max_length=True,
                    truncation=True)
        input_ids=torch.tensor(np.array(indices["input_ids"]))
        attention_masks=torch.tensor(np.array(indices["attention_mask"]), dtype=torch.long)

        return input_ids, labels, attention_masks
       
    def setup(self, stage=None):
        self.train_inputs, self.train_labels, self.train_masks = prepare_data(self.train_df)
        self.valid_inputs, self.valid_labels, self.valid_masks = prepare_data(self.valid_df)
        
    def train_dataloader(self) -> DataLoader:
        train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
        train_sampler = RandomSampler(train_data)
        return DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
    
    def val_dataloader(self) -> DataLoader:
        validation_data = TensorDataset(self.valid_inputs, self.valid_masks, self.valid_labels)
        validation_sampler = SequentialSampler(validation_data)
        return DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)
        
