import math

#import matplotlib.pyplot as plt 
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from torchvision.transforms import ToTensor

import PIL
import tqdm

# fuck it...
DEVICE = 'cuda'

class LinearSinus(nn.Module):
    def __init__(self, in_features, out_features, omega=30, is_first_layer=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega=float(omega)
        self.is_first_layer = is_first_layer
        
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def forward(self, x):
        x = x*self.omega
        x = self.linear(x)
        x = x.sin()
        return x
    
    def _init_weights(self):
        with torch.no_grad():
            k = math.sqrt(6/self.in_features)
            self._k0 = k
            if not self.is_first_layer:
                k = k / self.omega
            self._k1 = k
            self.linear.weight.uniform_(-k,k)

def make_idx(im_in):
    #idx = []
    #x,y = im_in.shape
    #for i in range(x):
    #    for j in range(y):
    #        idx.append((i,j))
    #idx = torch.tensor(idx, device=DEVICE)
    #return idx
    return make_idx_from_shape(im_in.shape)['coords']

def make_idx_from_shape(shape):
    x,y = shape
    idx = []
    for i in range(x):
        for j in range(y):
            idx.append((i,j))
    idx = torch.tensor(idx, device=DEVICE)
    return {'coords':idx, 'coords_rescaled':rescale_coords(idx, shape)}

def rescale_coords(coords, shape):
    scale = torch.tensor(shape, device=DEVICE)
    b = (coords / scale)
    b = (b-.5)/.5
    return b
        
class SirenImageLearner(pl.LightningModule):
    def __init__(self, 
                 in_features=2,
                 out_features=1,
                 dim_hidden=256,
                 n_hidden=5,
                 lr=1e-4):
        super().__init__()
        
        # getting an error complaining that self.hparams isn't a dict...
        #self.save_hyperparameters() # looks like this doesn't help
        
        #self.target_image = target_image
        self.in_features = in_features
        self.out_features = out_features
        self.dim_hidden = dim_hidden
        self.n_hidden = n_hidden
        self.lr = lr
        
        layers = [
            LinearSinus(
                in_features=in_features,
                out_features=dim_hidden,
                is_first_layer=True
            )]
        for _ in range(n_hidden):
            layers.append(LinearSinus(in_features=dim_hidden, out_features=dim_hidden))
        layers.append(LinearSinus(in_features=dim_hidden, out_features=out_features))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    
    def training_step(self, batch, batch_idx):
        tb = self.trainer.logger.experiment

        y_true = batch['pixel_values'].clone().detach().requires_grad_(True)
        coords_rescaled = batch['coords_rescaled'].clone().detach().requires_grad_(True)
        
        y_pred = self.forward(coords_rescaled).squeeze()
        loss = F.mse_loss(y_pred, y_true)
        
        #tb.add_histogram('loss_grad', loss.grad, batch_idx) # uh....
        self.log("train_loss", loss)
        
        # fuck it...
        idx0=batch['coords']
        im_recons=torch.sparse_coo_tensor(
            indices=idx0.T, 
            values=y_true,
            size=tuple(idx0.max(dim=0).values+1),
            device=torch.device(DEVICE) #device=DEVICE
        ).to_dense().unsqueeze(0)
        
        tb.add_image('source',im_recons, batch_idx)
        
        im_pred = torch.sparse_coo_tensor(
            indices=idx0.T, 
            values=y_pred,
            size=tuple(idx0.max(dim=0).values+1),
            device=torch.device(DEVICE) #device=DEVICE
        ).to_dense().unsqueeze(0)
        
        #tb.add_image('pred', im_pred, batch_idx)
        #tb.add_image('pred', im_pred)
        tb.add_image('pred', im_pred, self.global_step)
        
        # For debugging the nan loss
        #self.log('y_pred.shape', torch.tensor(y_pred.shape), batch_idx)
        #self.log('y_true.shape', torch.tensor(y_true.shape), batch_idx)
        
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SirenImageDataWrapper(pl.LightningDataModule):
    def __init__(self, target_image):
        super().__init__()
        self.target_image = target_image
    
    def prepare_data(self):
        self._scale = torch.tensor(self.target_image.shape, device=DEVICE) - 1
        coords = make_idx(self.target_image)
        self._coords = coords
        self._coords_rescaled = self._rescale_coords(coords)
        
        x = self._coords[:,0]
        y = self._coords[:,1]
        batch= {
            'pixel_values': self.target_image[x, y],
            'coords_rescaled': self._coords_rescaled,
            'coords':self._coords,
            'tgt_img': self.target_image
        }
        self._dataset = batch
        
        class Loader(torch.utils.data.IterableDataset):
            def __iter__(_self):
                yield self._dataset
                
        self._Loader = Loader()
        
    def _rescale_coords(self, coords):
        """
        Implicit representation takes coordinates as input and outputs pixel value at that position of the image.
        To stabilize network training, we need to rescale these coordinates from [0,im_size]**2 to [-1,1]**2
        """
        with torch.no_grad(): # probably doesn't make a difference...
            b = (coords / self._scale)
            b = (b-.5)/.5
        return b
    
    def train_dataloader(self):
        return self._Loader
    
if __name__ == '__main__':
    im_path = '../data/google-photos-export/uncompressed/takeout-20210901T023707Z-001/Takeout/Google Photos/lithophane candidates/20210313_133327.jpg'
    im = PIL.Image.open(im_path)
    im3 = im.resize((122, 163)).convert('L').rotate(-90, expand=True)
    t_im3 = ToTensor()(im3).squeeze()
    
    model = SirenImageLearner()
    dm = SirenImageDataWrapper(target_image=t_im3)
    trainer = pl.Trainer(
        gpus=-1,
        max_steps=1000,
        log_every_n_steps=50
    )
    trainer.fit(model, dm)