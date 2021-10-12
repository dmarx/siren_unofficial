import math

import mlflow
import mlflow.pytorch
#import matplotlib.pyplot as plt 
#import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from torchvision.transforms import ToTensor, ToPILImage

from loguru import logger
import matplotlib.pyplot as plt
import PIL
import tqdm

from kornia.losses import psnr_loss, ssim_loss, total_variation, kl_div_loss_2d
# kornia losses all seem to presume same shape input. Lame. 
# let's approximate a distributional divergence by taking quantiles 
# and calculating the area between the quantile curves (i.e. sum over abs differences)

# fuck it...
DEVICE = 'cuda'

def q_q_distance(t0, t1, n_quantiles=100):
    """
    Definitely didn't invent this. Closely related to kolmogorov-smirnoff, Kuiper, and
    Cramer-von Mises statistics. Wasserstein?
    """
    q = torch.linspace(0,1,steps=n_quantiles+1, device=DEVICE)[:-1]
    p0 = t0.quantile(q)
    p1 = t1.quantile(q)
    #d = math.sum(math.abs(p0-p1))
    d = sum(abs(p0-p1))
    return d

# fuck it, let's just use cramer von mises
def cramer_von_mises_distance(t0, t1, n_quantiles=100):
    """
    n_quantiles is basically the resolution of the approximation. 
    More quantiles calculated, the better the approximation,
    i.e. the higher the resolution of the estimato.
    """
    q = torch.linspace(0,1,steps=n_quantiles+1, device=DEVICE)[:-1]
    p0 = t0.quantile(q)
    p1 = t1.quantile(q)
    #d = math.sum(math.abs(p0-p1))
    sse = (p0-p1).pow(2).mean() #.sum()
    return sse
    
    
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
                 lr=1e-4,
                 notebook_mode=False,
                 super_resolution_factor=10,
                 warmup_steps=500, # 
                 cvm_alpha = 1000
                ):
        super().__init__()
        
        # getting an error complaining that self.hparams isn't a dict...
        #self.save_hyperparameters() # looks like this doesn't help
        # improve lighting docs: save_hyperparameters should be referenced in hparam logging, and vice versa
        # - https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html#lightningmodule-hyperparameters
        # - https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#logging-hyperparameters
        
        #self.target_image = target_image
        self.in_features = in_features
        self.out_features = out_features
        self.dim_hidden = dim_hidden
        self.n_hidden = n_hidden
        self.lr = lr
        self.notebook_mode = notebook_mode
        self.super_resolution_factor = super_resolution_factor
        self.warmup_steps = warmup_steps
        self.cvm_alpha = cvm_alpha
        
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
        
        
    @property
    def example_input_array(self):
        """
        To compute fwd pass for tensorboard computation graph logging.
        
        - https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.tensorboard.html
        """
        return torch.ones(self.in_features)
    
    @property
    def default_hp_metric(self):
        return False
    
        
    def forward(self, x):
        return self.block(x)
    
    def build_image_from_coords(self, pixel_values, coords=None, record=False):
        
        if coords is None:
            idx0 = self._idx0
        else:
            idx0 = coords

        im_recons=torch.sparse_coo_tensor(
            indices=idx0.T, 
            values=pixel_values,
            size=tuple(idx0.max(dim=0).values+1),
            device=torch.device(DEVICE) #device=DEVICE
        ).to_dense().unsqueeze(0)
        
        if record:
            self._idx0 = coords#.clone().detach()
            #self._im_recons = im_recons#.clone().detach()
        
        return im_recons
    
    def training_step(self, batch, batch_idx):
        tb=self.trainer.logger.experiment

        y_true = batch['pixel_values'].clone().detach().requires_grad_(True)
        coords_rescaled = batch['coords_rescaled'].clone().detach().requires_grad_(True)
        
        y_pred = self.forward(coords_rescaled).squeeze()
        loss = F.mse_loss(y_pred, y_true)
        
        #tb.add_histogram('loss_grad', loss.grad, batch_idx) # uh....
        self.log("train_loss", loss)
        
        
        #if not hasattr(self, '_im_recons'):
        if not hasattr(self, '_idx0'):
            im_recons = self.build_image_from_coords(y_true, coords=batch['coords'], record=True)
            tb.add_image('source', im_recons, 0)
            del im_recons
            torch.cuda.empty_cache()
            
            idx0 = self._idx0
            sr_k = self.super_resolution_factor
            x,y = tuple(idx0.max(dim=0).values+1)
            shape_resolve = (sr_k*x, sr_k*y)
            idx_resolve = make_idx_from_shape(shape_resolve)
            #with torch.no_grad(): # Maybe this'll suppress the out of memory error?
            #    y_resolve = self.forward(idx_resolve['coords_rescaled']).squeeze()
            
            self._resolve_coords = idx_resolve['coords']#.clone()
            self._resolve_coords_rescaled = idx_resolve['coords_rescaled']#.clone()
            
            
        else:
            #im_recons = self._im_recons
            idx0 = self._idx0
        
        if self.global_step > self.warmup_steps: 
            y_resolve = self.forward(self._resolve_coords_rescaled).squeeze()
            d_up = cramer_von_mises_distance(y_resolve, y_true, n_quantiles=100)
            tb.add_scalar(f"cramer_von_mises_distance/super resolution - {sr_k}", d_up, self.global_step)
            # calculating it anyway, may as well log it..
        
            loss2 = loss + d_up / self.cvm_alpha
            tb.add_scalar(f"train_loss_plus", loss2, self.global_step)
        
        
        #sr_k = 10        
  

        
        
        #test1 = (self.global_step < 100) and (self.global_step % 10 == 0)
        #test2 = (self.global_step % 100 == 0)
        test1 = False
        test2 = self.global_step % self.trainer.log_every_n_steps == 0
        if test1 or test2:
            #idx0=batch['coords']
            


            #im_pred = torch.sparse_coo_tensor(
            #    indices=idx0.T, 
            #    values=y_pred,
            #    size=tuple(idx0.max(dim=0).values+1),
            #    #requires_grad=False,
            #    device=torch.device(DEVICE) #device=DEVICE
            #).to_dense().unsqueeze(0)
            #im_pred = self.build_image_from_coords(y_true, coords=batch['coords'])
            im_pred = self.build_image_from_coords(y_true)

            tb.add_image('pred', im_pred, self.global_step)
            if self.notebook_mode:
                plt.imshow(im_pred.detach().squeeze().cpu().numpy())
                plt.show()

            # what happens if we generate an image solely using the coordinates between the training points?
            with torch.no_grad():
                #shape_orig = tuple(idx0.max(dim=0).values+1)
                shape_impute = tuple(idx0.max(dim=0).values)
                idx_impute = make_idx_from_shape(shape_impute)
                y_impute = self.forward(idx_impute['coords_rescaled']).squeeze()

                im_impute = torch.sparse_coo_tensor(
                    indices = idx_impute['coords'].T, 
                    values = y_impute,
                    size = shape_impute,
                    device = torch.device(DEVICE) #device=DEVICE
                ).to_dense().unsqueeze(0)

            tb.add_image('impute', im_impute, self.global_step)

            # Super resolution!
            with torch.no_grad():
                sr_k = 10
                x,y = tuple(idx0.max(dim=0).values+1)
                shape_resolve = (sr_k*x, sr_k*y)
                idx_resolve = make_idx_from_shape(shape_resolve)
                y_resolve = self.forward(idx_resolve['coords_rescaled']).squeeze()

                im_resolve = torch.sparse_coo_tensor(
                    indices = idx_resolve['coords'].T, 
                    values = y_resolve,
                    size = shape_resolve,
                    device = torch.device(DEVICE) #device=DEVICE
                ).to_dense().unsqueeze(0)

            tb.add_image(f'super resolution - {sr_k}x ', im_resolve, self.global_step)
        
                        
            # psnr_loss, ssim_loss, total_variation
            tv_pred = total_variation(im_pred)
            #self.log("tv_pred", tv_pred) # doesn't seem to work right when only calling it when I need it...
            tb.add_scalar("tv_pred", tv_pred, self.global_step)
            
            tv_up_k = total_variation(im_resolve)
            #self.log(f"tv_super resolution - {sr_k}", tv_pred)
            tb.add_scalar(f"tv_super resolution - {sr_k}", tv_up_k, self.global_step)
            
            #psnr_pred = psnr_loss(im_pred, im_recons, max_val=1)
            #self.log(f"psnr_pred", psnr_pred)
            #tb.add_scalar("psnr_pred", psnr_pred, self.global_step)
            # psnr is basically just a rescaled MSE
            
            #kl_div_loss_2d(input, target, reduction='mean')
            
            # Needs input tensors to be same shape
            #psnr_up_k = psnr_loss(im_resolve, im_recons, max_val=1)
            #self.log(f"psnr/super resolution - {sr_k}", psnr_pred)
            
            #kl_pred = kl_div_loss_2d(im_pred.unsqueeze(0), im_recons.unsqueeze(0))
            #self.log("kl_pred", kl_pred)
            #tb.log_metrics({"kl_pred":kl_pred}, step=self.global_step)
            #tb.scalar("kl_pred", kl_pred, step=self.global_step)
            #tb.add_scalar("kl_pred", kl_pred, self.global_step)
            
            # Also needs input tensors to be same shape... blech.
            #kl_up_k = kl_div_loss_2d(im_resolve.unsqueeze(0), im_recons.unsqueeze(0))
            #self.log(f"kl/super resolution - {sr_k}", kl_up_k)
            
            #d_up = q_q_distance(im_resolve, im_recons, n_quantiles=100)
            #d_up = cramer_von_mises_distance(im_resolve, im_recons, n_quantiles=100)
            #tb.add_scalar(f"cramer_von_mises_distance/super resolution - {sr_k}", d_up, self.global_step)
        
        return loss
    
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=300)
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return {'optimizer':optimizer,
                'lr_scheduler':{
                    'scheduler':scheduler, # with scheduler, I def want LR logged...
                    "interval": "step",
                    }
               }


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
    #logger.debug(mlflow.get_artifact_uri())
    #mlflow_exprmnt_name = "SIREN-3"
    #mlflow_tracking_uri = 'sqlite:///../mlflow/mlflow.db'
    #mlflow.set_artifact_uri('file://../mlflow')
    #mlflow.set_tracking_uri(mlflow_tracking_uri)
    #mlflow.set_experiment(mlflow_exprmnt_name+"_outterlogger")
    #mlflow.pytorch.autolog() # this thing sucks. can't log images with this on.
    #logger.debug(mlflow.get_artifact_uri())

        
    im_path = '../data/google-photos-export/uncompressed/takeout-20210901T023707Z-001/Takeout/Google Photos/lithophane candidates/20210313_133327.jpg'
    im = PIL.Image.open(im_path)
    im3 = im.resize((122, 163)).convert('L').rotate(-90, expand=True)
    t_im3 = ToTensor()(im3).squeeze()
    
    from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger, WandbLogger

    #wandb_logger = WandbLogger()
    
    #logger1 = TensorBoardLogger("tb_logs", name="my_model")
    tb_logger = TensorBoardLogger("tb_logs", name="siren-upsample")
    #logger2 = MLFlowLogger(
    #        experiment_name=mlflow_exprmnt_name+"_sublogger", 
    #        tracking_uri=mlflow_tracking_uri,
    #        artifact_location='../mlflow/artifacts/'
    #)
    #loggers=[logger1, logger2]
    #loggers=[logger1]
    #trainer = Trainer(logger=[logger1, logger2])
    
    model = SirenImageLearner()
    dm = SirenImageDataWrapper(target_image=t_im3)
    
    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    trainer = pl.Trainer(
        gpus=-1,
        #max_steps=1000,
        max_steps=100000, # I think there's a way to use CLI args to override here
        log_every_n_steps=100, #50,
        #logger=loggers,
        logger=tb_logger,
        log_gpu_memory=True, # fuck yeah!
        weights_summary=None,
    )
        # shouldn't need to do this...
    #with mlflow.start_run():
    #with logger2.experiment.start_run():
    trainer.fit(model, dm)