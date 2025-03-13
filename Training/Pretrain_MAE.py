import os
import math
import logging
from omegaconf import DictConfig
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from Data import SamData, ImageNet
from models import MaskedAutoEncoder
from Configs import config 
from Training import Metrics

class Pretrainer(object):

    """
    Trainer class for self-supervised MAE pre-training
    """

    def __init__(self, args: DictConfig, train_transforms=None, test_transforms=None):

        self.args = args

        # Check image size
        if self.args.img_size % 16 != 0:
            raise Exception(f"Specified image size {self.args.img_size} is not divisible by 16.")
        
        # Get ViT Backbone params 
        self.embed_dim = config.VIT_DICT[args.vit_size]['embed_dim']
        self.encoder_depth = config.VIT_DICT[args.vit_size]['depth']
        self.encpder_heads = config.VIT_DICT[args.vit_size]['heads']

        # Init model 
        self.model = MaskedAutoEncoder.MAE_ViT(image_size=args.img_size,
                                               num_channels=1,
                                               patch_size=args.patch_size,
                                               emb_dim = self.embed_dim,
                                               encoder_layers=self.encoder_depth,
                                               encoder_heads=self.encpder_heads,
                                               mask_ratio=self.args.mask_ratio)
        
        # Init optimization logic  
        self.warmup_epochs = self.args.epochs * 0.1

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return float(epoch) / float(self.warmup_epochs)
            else:
                # Cosine annealing
                return 0.5 * (1 + math.cos((epoch - self.warmup_epochs) / (self.args.pretrain_epochs - self.warmup_epochs) * 3.141592653589793))
            
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)  
        self.loss = nn.MSELoss()
        
        # Init dataset with preprocessing
        if train_transforms == None or test_transforms==None:
            raise Exception('Transforms have not been properly initialized for pretraining.')
        self.test_transforms = test_transforms
        self.train_transforms = train_transforms
        self._create_dataloaders()

        # to cuda or cpu
        self._to_device()

        # init training utils
        self.model_dir = os.path.abspath(os.path.join('Training',
                                                      'models',
                                                      'pretrain',
                                                      self.args.dataset, 
                                                      self.args.pretrain_mode,
                                                      'ViT-' + (self.args.vit_size).upper(),  
                                                      f"{self.args.img_size}x{self.args.img_size}_{self.args.patch_size}x{self.args.patch_size}_augmentations_{self.args.augmentations}_" + 
                                                      datetime.now().strftime("%m%d-%H%M")))

        # Make logdir
        self.log_dir = os.path.join(self.model_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        print('Writing logs to dir: ' + self.log_dir)

        # Make checkpoint dir
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('Saving checkpoints in: ' + self.checkpoint_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.epoch = 1


    def _to_device(self): 
        device = self.args.device if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.loss.to(device)


    def _create_dataloaders(self): 
 
        # get train and test datasets
        match self.args.dataset: 
            case 'imagenet':
                dataset_train = ImageNet.ImageNet100(img_size=self.args.img_size, split='train', transform=self.train_transforms)
                dataset_test = ImageNet.ImageNet100(img_size=self.args.img_size, split='test', transform=self.test_transforms)
                print('using imagenet! YAY')
            case 'sam':
                dataset_train = SamData.get_sam_dataset(mode='pretrain', img_size=self.args.img_size, transforms=self.train_transforms, split='train')
                dataset_test = SamData.get_sam_dataset(mode='pretrain', img_size=self.args.img_size, transforms=self.test_transforms, split='test')


        # create dataloaders 
        self.dataloader_train = DataLoader(dataset_train, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory = self.args.pin_memory)
        self.dataloader_test = DataLoader(dataset_test, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory = self.args.pin_memory)

        
    def train(self):
        print(f'STARTING PRETRAINING WITH {self.args.pretrain_mode.upper()}')

        params = sum(p.numel() for p in self.model.parameters())
        print(f'PARAMETERS OF PRE-TRAINING MODEL: {params}')

        for epoch in tqdm(range(self.args.pretrain_epochs)): 
            self.epoch = epoch
            training_logs, _, test_metrics, _ = self._train_test_one_epoch()
            self.log_metrics(epoch, self.args.pretrain_epochs, training_logs)

            checkpoint_frequency = 100 if self.args.dataset == 'sam' else 50

            if (self.epoch + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(loss = test_metrics['loss'],
                                      mse = test_metrics['mse'], 
                                      ssim = test_metrics['ssim'],
                                      psnr = test_metrics['psnr'])

        self.writer.close()
        print("PRETRAINING ENDED")
        
        return self.model


    def _train_test_one_epoch(self): 

        device = self.args.device if torch.cuda.is_available() else "cpu"

        """
        start training loop with parameters intialized 
        """
        
        running_loss_train = 0.0
        running_loss_test = 0.0
        
        running_mse_train = 0.0
        running_ssim_train = 0.0
        running_psnr_train = 0.0
        running_mse_test = 0.0
        running_ssim_test = 0.0
        running_psnr_test = 0.0

        print("-------  TRAINING  -------")

        idx_loss = 0 
        with tqdm(self.dataloader_train) as iterator:
            
            for imgs in iterator:
                imgs = imgs.to(device)

                loss, outputs = self._train_one_batch(imgs)
                running_loss_train += loss.cpu().detach().item()

                # Get and log train evaluation metrics 
                mse, ssim, psnr = Metrics.get_evaluation_metrics(imgs, outputs, device)
                
                running_mse_train += mse.cpu().detach().item()
                running_ssim_train += ssim.cpu().detach().item()
                running_psnr_train += psnr.cpu().detach().item()
                
                self.writer.add_scalar("Running Training Loss", loss.cpu().detach().item(), idx_loss)

                idx_loss += 1
            

        print("-------  TESTING  -------")
        with tqdm(self.dataloader_test) as iterator: 
            idx_loss = 0 
            for imgs in iterator: 

                imgs = imgs.to(device)

                loss, outputs = self._test_one_batch(imgs)
                running_loss_test += loss.cpu().detach().item()

                # Get and log test evaluation metrics 
                mse, ssim, psnr = Metrics.get_evaluation_metrics(imgs, outputs, device)
                running_mse_test += mse.cpu().detach().item()
                running_ssim_test += ssim.cpu().detach().item()
                running_psnr_test += psnr.cpu().detach().item()
                
                self.writer.add_scalar("Running Test Loss", loss.cpu().detach().item(), idx_loss)
                idx_loss += 1

        self.writer.add_scalar("Epoch Training Loss", running_loss_train/len(self.dataloader_train), self.epoch)
        self.writer.add_scalar("Epoch Testing Loss", running_loss_test/len(self.dataloader_test), self.epoch)
        self.writer.add_scalar("Epoch Training MSE", running_mse_train/len(self.dataloader_train), self.epoch)
        self.writer.add_scalar("Epoch Testing MSE", running_mse_test/len(self.dataloader_test), self.epoch)
        self.writer.add_scalar("Epoch Training SSIM", running_ssim_train/len(self.dataloader_train), self.epoch)
        self.writer.add_scalar("Epoch Testing SSIM", running_ssim_test/len(self.dataloader_test), self.epoch)
        self.writer.add_scalar("Epoch Training PSNR", running_psnr_train/len(self.dataloader_train), self.epoch)
        self.writer.add_scalar("Epoch Testing PSNR", running_psnr_test/len(self.dataloader_test), self.epoch)
        
        logs = {"train_loss": running_loss_train/len(self.dataloader_train), 
            "test_loss": running_loss_test/len(self.dataloader_test), 
            "train_mse": running_mse_train/len(self.dataloader_train), 
            "train_ssim": running_ssim_train/len(self.dataloader_train), 
            "train_psnr": running_psnr_train/len(self.dataloader_train), 
            "test_mse": running_mse_test/len(self.dataloader_test), 
            "test_ssim": running_ssim_test/len(self.dataloader_test), 
            "test_psnr": running_psnr_test/len(self.dataloader_test)}
        test_metrics = {'loss': running_loss_test/len(self.dataloader_test),
                    'mse': running_mse_test/len(self.dataloader_test), 
                    'ssim': running_ssim_test/len(self.dataloader_test),
                    'psnr': running_psnr_test/len(self.dataloader_test)}

        return logs, False, test_metrics, True


    def _train_one_batch(self, imgs): 
            
        # Set model in training mode
        self.model.train()

        self.optimizer.zero_grad()

        # Forward pass
        model_output, mask_img, _ = self.model(imgs) 

        # Invert mask to compute reconstruction loss only on masked patches 
        inverted_mask = (((-1) * mask_img) + 1).detach()

        # Train model 
        loss = self.loss(model_output * inverted_mask, imgs * inverted_mask)

        loss.backward()  
        self.optimizer.step() 

        self.scheduler.step()

        return loss, model_output
    

    def _test_one_batch(self, imgs): 

        # Set model in eval mode
        self.model.eval()

        # Forward pass
        model_output, mask_img, _ = self.model(imgs) 

        # Invert mask to compute reconstruction loss only on masked patches 
        inverted_mask = (((-1) * mask_img) + 1)

        # Test model 
        loss = self.loss(model_output * inverted_mask, imgs * inverted_mask)
        
        return loss, model_output
    

    def _save_checkpoint(self, loss, mse, ssim, psnr):

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pth")
        torch.save({
            'epoch': self.epoch,
            'model': self.model,
            'loss': loss,
            'mse': mse,
            'ssim': ssim, 
            'psnr': psnr
        }, checkpoint_path)

        print(f'SAVED CHECKPOINT FOR EPOCH: {self.epoch}')


    def log_metrics(self, epoch, epochs, logs):
        logging.info(f"EPOCH [{epoch}/{epochs}]")
        logging.info(f"""
                        LOSS = [{logs['train_loss']:.4f}/{logs['test_loss']:.4f}]""")
