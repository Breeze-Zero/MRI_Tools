'''
MIT License
Copyright (c) 2023 Breeze.
'''
import torch
import pytorch_lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from NAFNet import NAFNet
from STN import SpatialTransformer,gradient_loss
from timm.scheduler.cosine_lr import CosineLRScheduler
from losses import L1_Charbonnier_loss,LCC
from torch.utils.data import DataLoader    
from datasets import Build_dataset
import wandb
from pytorch_lightning.loggers import WandbLogger
import argparse
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
# from torchmetrics.functional import dice as DICE
from Metric import DistributedMetricSum

class LightningModel(LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.DenoiseModel = NAFNet(img_channel=1, width=32, middle_blk_num=12,
                  enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
        self.RegistrationModel = SpatialTransformer(channels=1)
        self.lr = self.args.lr
        self.gradient_loss = gradient_loss
        self.similarity = LCC()
        self.selfloss = L1_Charbonnier_loss()#torch.nn.MSELoss()#
        self.crossloss = L1_Charbonnier_loss()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.train_dataset,self.val_dataset = Build_dataset(root_path=args.path_root) ## M4raw old version need to change
        self.Metrics_ssim = DistributedMetricSum() 
        self.Metrics_psnr = DistributedMetricSum()
        self.Metrics_lcc = DistributedMetricSum()
        # self.show_tag = True
    def forward(self, x,ref):
        x = self.DenoiseModel(x)
        grid,offset = self.RegistrationModel(x,ref)
        x = self.RegistrationModel.warp(x,grid)
        return x

    def training_step(self, batch, batch_idx):
        D_opt, R_opt = self.optimizers()
        x_noise, x_clean,y_noise,y_clean = batch
                
        x_out = self.DenoiseModel(x_noise)
        y_out = self.DenoiseModel(y_noise)
        selfloss = self.selfloss(x_out,x_clean) + self.selfloss(y_out,y_clean)


        grid_y, offset_y = self.RegistrationModel(y_clean,x_clean)
        y_reg = self.RegistrationModel.warp(y_clean,grid_y)


        grid_x, offset_x = self.RegistrationModel(x_clean, y_clean)
        x_reg = self.RegistrationModel.warp(x_clean, grid_x)
        
        crossloss = self.crossloss(x_out,y_reg.detach())  + self.crossloss(y_out,x_reg.detach())
        ##########################
        # Optimize DenoiseModel #
        ########################## 
        D_loss = 2 * selfloss + crossloss
        D_opt.zero_grad()
        self.manual_backward(D_loss,retain_graph=True)    

        similarity_loss = self.similarity(y_reg,x_clean) + self.similarity(x_reg,y_clean)
        smooth_loss = self.gradient_loss(offset_y) + self.gradient_loss(offset_x)
        
        ######################
        # Optimize RegistrationModel #
        ######################
        R_loss = similarity_loss + 1000 * smooth_loss
        R_opt.zero_grad()
        self.manual_backward(R_loss,retain_graph=True)
        D_opt.step() 
        R_opt.step()    ##这个BUG难了我两个钟，特此记录一下
 
        
        if self.global_step%1000==0:
            x_n = torch.einsum('chw->hwc', x_noise[0].detach().cpu()).numpy()
            x_b = torch.einsum('chw->hwc', x_clean[0].detach().cpu()).numpy()
            x_o = torch.einsum('chw->hwc', x_out[0].detach().cpu()).numpy()

            y_b = torch.einsum('chw->hwc', y_noise[0].detach().cpu()).numpy()
            y_o = torch.einsum('chw->hwc', y_out[0].detach().cpu()).numpy()
            y_r = torch.einsum('chw->hwc', y_reg[0].detach().cpu()).numpy()

            self.logger.experiment.log({"x_noise":wandb.Image(x_n, caption="x_noise"),
                                        "x_target":wandb.Image(x_b, caption="x_target"),
                                        "x_out":wandb.Image(x_o, caption="x_out"),
                                        "y_base":wandb.Image(y_b, caption="y_base"),
                                        "y_out":wandb.Image(y_o, caption="y_out"),
                                        "y_reg":wandb.Image(y_r, caption="y_reg")
                                       })
                   
        self.log_dict({"D_loss": D_loss, "R_loss": R_loss}, prog_bar=True)
        
    def validation_step(self, batch, batch_idx):
        x_noise, x_clean,y_clean = batch

        grid_y, offset_y = self.RegistrationModel(y_clean,x_clean)
        y_reg = self.RegistrationModel.warp(y_clean,grid_y)
        lcc = 1-self.similarity(y_reg,x_clean)
        
        targets = (y_reg + x_clean)/2
        
        output = self.DenoiseModel(x_noise)
        
        ssim = SSIM(output,targets,data_range = 1.0)*lcc
        psnr = PSNR(output,targets,data_range = 1.0)*lcc
        self.Metrics_ssim(ssim)
        self.Metrics_psnr(psnr)
        self.Metrics_lcc(lcc)
        
#         if self.show_tag:
#             x_n = torch.einsum('chw->hwc', x_noise[0].detach()).cpu().numpy()
#             x_b = torch.einsum('chw->hwc', x_clean[0].detach()).cpu().numpy()
#             x_o = torch.einsum('chw->hwc', output[0].detach()).cpu().numpy()

#             y_b = torch.einsum('chw->hwc', targets[0].detach()).cpu().numpy()
#             y_r = torch.einsum('chw->hwc', y_reg[0].detach()).cpu().numpy()
#             y_i = torch.einsum('chw->hwc', y_noise[0].detach()).cpu().numpy()
#             self.logger.experiment.log({"x_noise":wandb.Image(x_n, caption="x_noise"),
#                                         "x_base":wandb.Image(x_b, caption="x_base"),
#                                         "x_out":wandb.Image(x_o, caption="x_out"),
#                                         "target":wandb.Image(y_b, caption="target"),
#                                         "y_base":wandb.Image(y_i, caption="y_base"),
#                                         "y_reg":wandb.Image(y_r, caption="y_reg")
#                                        })
#             self.show_tag = False
        

        
    def validation_epoch_end(self, outputs):
        ssim = self.Metrics_ssim.compute()
        psnr = self.Metrics_psnr.compute()
        lcc = self.Metrics_lcc.compute()
        self.log_dict({"SSIM": ssim,'PSNR':psnr,'LCC':lcc},prog_bar=True,sync_dist=True)
 
        self.Metrics_ssim.reset()
        self.Metrics_psnr.reset()
        self.Metrics_lcc.reset()
        # self.show_tag = True

    def configure_optimizers(self):
        D_opt = torch.optim.AdamW(self.DenoiseModel.parameters(), self.lr,weight_decay=0.,betas=(0.9, 0.9))
        R_opt = torch.optim.AdamW(self.RegistrationModel.parameters(), self.lr,weight_decay=0.,betas=(0.9, 0.9))
        D_lr_scheduler = CosineLRScheduler(
            D_opt,
            t_initial=self.args.num_epochs,
            cycle_mul=1.,
            lr_min=1e-7,
            warmup_lr_init=5e-7,
            warmup_t=self.args.warmup_epoch,
            cycle_limit=1,
            t_in_epochs=True,
        )
        R_lr_scheduler = CosineLRScheduler(
            R_opt,
            t_initial=self.args.num_epochs,
            cycle_mul=1.,
            lr_min=1e-7,
            warmup_lr_init=5e-7,
            warmup_t=self.args.warmup_epoch,
            cycle_limit=1,
            t_in_epochs=True,
        )
        
        return [D_opt, R_opt], [D_lr_scheduler,R_lr_scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        pass
    
    def training_epoch_end(self, outputs):
        D_lr_scheduler,R_lr_scheduler = self.lr_schedulers()
        D_lr_scheduler.step(self.current_epoch)
        R_lr_scheduler.step(self.current_epoch)
        
        
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.train_batchsize, shuffle=True, num_workers=self.args.num_workers,drop_last=True,pin_memory=self.args.pin_memory)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.val_batchsize, shuffle=False, num_workers=self.args.num_workers,drop_last=False,pin_memory=self.args.pin_memory)
        return val_loader
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train denoise with pytorch-lightning')

    parser.add_argument('--model_name',default='Denoise',type=str)
    parser.add_argument('--path_root',default='/data0/M4RawV1.5',type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_num', default=2, type=int)
    parser.add_argument('--use_pretrain', default=False, type=bool)
    parser.add_argument('--batch_dice', default=False, type=bool)
    parser.add_argument('--pin_memory', default=True, type=bool) 
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs of training.')
    parser.add_argument('--warmup_epoch', default=2, type=int)
    parser.add_argument('--train_batchsize', default=1, type=int)
    parser.add_argument('--val_batchsize', default=4, type=int)
    parser.add_argument('--sw_batchsize', default=4, type=int)
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_interval', default=50, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--checkpoint_path',default='',type=str)
    parser.add_argument('--resume',default=None,type=str)
    parser.add_argument('--wandbID',default=None,type=str)
    parser.add_argument('--find_unused_parameters', default=False, type=bool)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

    args = parser.parse_args()
    if args.resume is not None:
        wandb_logger = WandbLogger(project="Denoise",resume='allow',id=args.wandbID,)
    else:
        wandb_logger = WandbLogger(project="Denoise",name =args.model_name,)
 
    pytorch_lightning.utilities.seed.seed_everything(seed=args.seed, workers=True)
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="PSNR",
        mode="max",
        dirpath="./checkpoint/",
        filename=f"{args.model_name}-best_metric_model",
    )
    current_checkpoint_callback = ModelCheckpoint(dirpath="./checkpoint/",
                filename='{epoch}',every_n_epochs=5,
                        save_on_train_epoch_end =True)
    from pytorch_lightning.callbacks import ModelSummary
    model = LightningModel(args)
    if args.gpu_num >1:
        from pytorch_lightning.strategies import DDPStrategy
#         from pytorch_lightning.profilers import PyTorchProfiler

#         profiler = PyTorchProfiler(filename="perf-logs")
        trainer = Trainer(max_epochs=args.num_epochs,
                          devices=args.gpu_num,
                          # gpus=args.gpu_num,
                          accelerator="gpu",
                          strategy=DDPStrategy(find_unused_parameters=args.find_unused_parameters),
                          #strategy=DDPStrategy(gradient_as_bucket_view=True),#'ddp_find_unused_parameters_false',#
                          callbacks=[current_checkpoint_callback,best_checkpoint_callback,ModelSummary(max_depth=3)],
                          logger=wandb_logger,
                          check_val_every_n_epoch=args.val_interval,
                          precision=32,
                          log_every_n_steps=10,
                            sync_batchnorm=True,
                          num_sanity_val_steps=0,
                          detect_anomaly=False,
                          # limit_train_batches=1,
                          # limit_val_batches=1,
                          #profiler=profiler
                         )
    else:
        trainer = Trainer(max_epochs=args.num_epochs,
                          devices=args.gpu_num,
                          # gpus=args.gpu_num,
                          accelerator="gpu",
                          callbacks=[current_checkpoint_callback,best_checkpoint_callback,ModelSummary(max_depth=3)],
                          logger=wandb_logger,
                          check_val_every_n_epoch=args.val_interval,
                          log_every_n_steps=20,
                          num_sanity_val_steps=0,
                        #   auto_lr_find='lr',
                          precision=32,
                          detect_anomaly=False,
                          # profiler="simple",
                          # limit_train_batches=5,
                          # limit_val_batches=1,
                         )

        # trainer.tune(model)

    trainer.fit(model,ckpt_path=args.resume)
