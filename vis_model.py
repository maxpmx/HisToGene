import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformer import ViT
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet101'):
        super(FeatureExtractor, self).__init__()
        backbone = torchvision.models.resnet101(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        # self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        return x

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, backbone='resnet50', learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        backbone = torchvision.models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = num_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)
        # self.valid_acc = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.feature_extractor(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        return parser


class STModel(pl.LightningModule):
    def __init__(self, feature_model=None, n_genes=1000, hidden_dim=2048, learning_rate=1e-5, use_mask=False, use_pos=False, cls=False):
        super().__init__()
        self.save_hyperparameters()
        # self.feature_model = None
        if feature_model:
            # self.feature_model = ImageClassifier.load_from_checkpoint(feature_model)
            # self.feature_model.freeze()
            self.feature_extractor = ImageClassifier.load_from_checkpoint(feature_model)
        else:
            self.feature_extractor = FeatureExtractor()
        # self.pos_embed = nn.Linear(2, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch, center):
        feature = self.feature_extractor(patch).flatten(1)
        h = feature
        pred = self.pred_head(F.relu(h))
        return pred

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('valid_loss', loss)
        
    def test_step(self, batch, batch_idx):
        patch, center, exp, mask, label = batch
        if self.use_mask:
            pred, mask_pred = self(patch, center)
        else:
            pred = self(patch, center)

        loss = F.mse_loss(pred, exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


class SpatialTransformer(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=64):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        patch_dim = 3*patch_size*patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):        
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    a = torch.rand(1,4000,3*112*112)
    p = torch.ones(1,4000,2).long()
    model = SpatialTransformer()
    print(count_parameters(model))
    x = model(a,p)
    print(x.shape)
