"""
Models module within the Brigit package.

Contains the architecture of the models used
for the project.

Copyright by Raúl Fernández Díaz
"""

import torch
import numpy as np
import sklearn.metrics as skmetrics
from torch import nn
import pytorch_lightning as pl
import torchmetrics.functional as metrics
from .layers import conv3D, linear


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        predictions = self(images)
        loss = self.loss(predictions, labels.float())

        return {
            'labels': labels,
            'predictions': predictions,
            'loss': loss
        }

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        loss = self.loss(predictions, labels.float())

        return {
            'val_labels': labels,
            'val_predictions': predictions,
            'val_loss': loss
        }

    def training_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs])
        predictions = torch.cat([x['predictions'] for x in outputs])
        loss = self.loss(predictions, labels.float())
        labels = labels.int()
        acc = metrics.accuracy(predictions, labels)

        self.log("train_acc", acc)
        self.log("train_loss", loss)

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x['val_labels'] for x in outputs])
        predictions = torch.cat([x['val_predictions'] for x in outputs])
        loss = self.loss(predictions, labels.float())
        labels = labels.int()
        acc = metrics.accuracy(predictions, labels)
        prec, rec = metrics.precision_recall(predictions, labels,
                                             average='micro')
        f1 = metrics.f1_score(predictions, labels, average='micro')
        spe = metrics.specificity(predictions, labels, average='micro')
        auroc = metrics.auroc(predictions, labels, num_classes=1,
                              average='micro')

        labels = labels.detach().cpu().numpy()
        predictions = (predictions > 0.5).int().detach().cpu().numpy()
        cohen = torch.from_numpy(
            np.array(skmetrics.cohen_kappa_score(labels, predictions))
        )
        mcc = torch.from_numpy(
            np.array(skmetrics.matthews_corrcoef(labels, predictions))
        )
        log_loss = torch.from_numpy(
            np.array(skmetrics.log_loss(labels, predictions))
        )

        self.log("acc", acc)
        self.log("loss", loss)
        self.log("f1", f1)
        self.log("auroc", auroc)
        self.log("precision", prec)
        self.log("recall", rec)
        self.log("specificity", spe)
        self.log("cohen", cohen)
        self.log("mcc", mcc)
        self.log("log_loss", log_loss)


class BrigitCNN(BaseModel):
    def __init__(
        self,
        learning_rate: float,
        neurons_layer: int = 32,
        size: int = 12,
        num_dimns: int = 6,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters(
            "learning_rate",
            "neurons_layer",
            "num_dimns",
        )
        self.learning_rate = learning_rate
        self.example_input_array = torch.rand(
            num_dimns*size**3, 1).view(1, num_dimns, size, size, size)

        self.convolutional = nn.Sequential(
            nn.Dropout3d(0.2),
            conv3D(num_dimns, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            conv3D(neurons_layer, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            nn.BatchNorm3d(neurons_layer),
            conv3D(neurons_layer, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            conv3D(neurons_layer, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            nn.BatchNorm3d(neurons_layer),
            nn.AvgPool3d(2),
            conv3D(neurons_layer, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            conv3D(neurons_layer, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            nn.BatchNorm3d(neurons_layer),
            conv3D(neurons_layer, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            conv3D(neurons_layer, neurons_layer, 3, 1, nn.LeakyReLU, 0.2),
            nn.BatchNorm3d(neurons_layer),
            nn.AvgPool3d(2),
            nn.Flatten()
        )
        self.classifier1 = nn.Sequential(
            linear(neurons_layer*3**3, 512, 0.2, nn.LeakyReLU),
            linear(512, 256, 0.2, nn.LeakyReLU)
        )
        self.classifier2 = nn.Sequential(
            linear(256, 1, 0.0, nn.Sigmoid)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = self.classifier1(x)
        return self.classifier2(x)


if __name__ == '__main__':
    help(BaseModel)
    help(BrigitCNN)
