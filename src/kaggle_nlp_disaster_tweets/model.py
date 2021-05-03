import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import F1
from transformers import AdamW, get_linear_schedule_with_warmup

class LitNLPModel(LightningModule):
    def __init__(self, 
                 model, 
                 epochs,
                 lr: float = 6e-6,
                 warmup: int = 0):
      
        super().__init__()
        
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup
        
        self.f1_score = F1(num_classes=2)
        
    def forward(self, b_input_ids, b_input_mask, b_labels):
        output = self.model(b_input_ids, 
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
        optimizer = AdamW(model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=self.warmup, 
                                            num_training_steps=189*self.epochs)
        return [optimizer], [scheduler]
    
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
