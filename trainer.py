import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class PLModel(pl.LightningModule):
    def __init__(self, classes):
        super(PLModel, self).__init__()
        self.model =  mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.model.classifier[3] = nn.Linear(1024, len(classes))
        self.xentropy = nn.CrossEntropyLoss()
        self.train_loss,self.val_loss,self.test_loss = 0.0, 0.0, 0.0
        self.train_acc,self.val_acc,self.test_acc = 0,0,0

    def forward(self, x):
        out = self.model(x)  # x : [B,C,H,W]  --> out : [B,n,H,W]
        return out

    def loss(self, y_hat, y):
        loss = self.xentropy(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)  # this calls self.forward
        loss = self.loss(logits, labels)

        pred = torch.argmax(logits,dim=1)
        matched = pred == labels
        accu = matched.sum()
        self.train_acc += accu.item()
        accu = accu.item() / len(labels)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar= True)
        self.log('train_accu_curr', accu, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train_epoch_accu', self.train_acc, on_step=False, on_epoch=True)
        self.train_acc = 0.0

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        pred = torch.argmax(logits, dim=1)
        matched = pred == labels
        accu = matched.sum()
        self.val_acc += accu.item()
        accu = accu.item() / len(labels)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_accu_curr', accu, on_step=True, on_epoch=False, prog_bar=True)
        return {'val_acc': self.val_acc}

    def validation_epoch_end(self, outputs):
        self.log('val_epoch_accu', self.val_acc, on_step=False, on_epoch=True)
        self.val_acc = 0.0

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        pred = torch.argmax(logits, dim=1)
        matched = pred == labels
        accu = matched.sum()
        self.test_acc += accu.item()
        accu = accu.item() / len(labels)

        self.log('test_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('test_accu_curr', accu, on_step=True, on_epoch=False, prog_bar=True)
        return {'test_acc': self.test_acc}

    def test_epoch_end(self, test_step_outputs):
        self.log('test_epoch_accu', self.test_acc, on_step=False, on_epoch=True)
        self.test_acc = 0.0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer]