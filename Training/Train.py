import os 
import sys
import logging
from datetime import datetime
from omegaconf import DictConfig
from tqdm import tqdm 

import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
import torchvision.models as models 

from models import ViTClassifier, SAM_CNN, PCB_CNN
from Training import Pretrain_MAE
from Data import SamData
from Configs import config 

logging.basicConfig(level=logging.INFO)

class Trainer(object):

    """
    Trainer class for supervised Fault Detection Training
    """


    def __init__(self, cfg: DictConfig): 
        
        self.args = cfg
        
        # check image size
        if self.args.img_size % 16 != 0:
            raise Exception(f"Specified image size {self.args.img_size} is not divisible by 16.")
        
        self.loss = nn.MSELoss()
        self.min_mse = float('inf')

        # init transforms 
        self._init_transforms()

        # PRETRAINING
        pretrained_model = None
        
        if self.args.use_pretrained_model or self.args.pretrained_model != 'None' or self.args.only_pretrain:

            # if available, load pretrained model 
            if self.args.pretrained_model != 'None':
                print('USING PRETRAINED MODEL FROM DIR: ' + str(self.args.pretrained_model))
                pretrained_model = torch.load(self.args.pretrained_model, map_location=self.args.device, weights_only=False)['model']

            # else pretrain a model from scratch
            else: 
                pretrainer = Pretrain_MAE.Pretrainer(args=self.args, train_transforms=self.train_transforms, test_transforms=self.test_transforms) 
              
                # start pretraining
                pretrained_model = pretrainer.train()

                if self.args.only_pretrain:
                    sys.exit(0)
                else:
                    print('PRETRAINING MODEL FROM SCRATCH')

        # init model
        if cfg.use_baseline: 
            match cfg.baseline_model: 
                case 'resnet50':
                    self.model = models.resnet50()
                    num_features = self.model.fc.in_features
                    self.model.fc = torch.nn.Linear(num_features, 1) 
                case 'efficientnetb7':
                    self.model = models.efficientnet_b7()
                    in_features = self.model.classifier[1].in_features  
                    self.model.classifier = nn.Sequential(
                        nn.Dropout(p=0.2, inplace=True),  
                        nn.Linear(in_features=in_features, out_features=1) 
                    )
                case 'vgg11':
                    self.model = models.vgg11()
                    self.model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
                case 'sam_cnn': 
                    self.model = SAM_CNN.SAM_CNN(num_classes=1)
                case 'pcb_cnn':
                    self.model = PCB_CNN.PCB_CNN()
      
        else: 
            self.model = ViTClassifier.ViT(num_classes=0, 
                                            image_size=self.args.img_size, 
                                            num_channels=1 if self.args.dataset in ['sam', 'mnist'] else 3,
                                            patch_size=self.args.patch_size, 
                                            emb_dim=config.VIT_DICT[self.args.vit_size]['embed_dim'],
                                            encoder_layer=config.VIT_DICT[self.args.vit_size]['depth'],
                                            encoder_head=config.VIT_DICT[self.args.vit_size]['heads'],
                                            mode=self.args.mode,
                                            pretrained_model=pretrained_model)

        # init training params 
        if self.args.classification_mode == 'end2end':
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr)
        elif self.args.classification_mode == 'linear':
            self.optimizer = torch.optim.AdamW(params=self.model.classifier.parameters(), lr=self.args.lr)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.pretrain_epochs)  

        # init dataloaders 
        self._create_dataloaders()

        # to cuda or cpu
        self._to_device()

        # init training utils
        self.model_dir = os.path.abspath(os.path.join('Training',
                                                      'models',
                                                      'downstream',
                                                      self.args.dataset, 
                                                      self.args.mode, 
                                                      f"{self.args.img_size}x{self.args.img_size}_{self.args.patch_size}x{self.args.patch_size}_augmentations_{self.args.augmentations}_" + datetime.now().strftime("%m%d-%H%M")))

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

        # Make final model dir
        self.final_model_dir = os.path.join(self.model_dir, 'final model')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('Saving final model in: ' + self.final_model_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.epoch = 0 


    @property
    def parameters(self): 
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    
    def train(self): 
        for epoch in tqdm(range(self.args.epochs)): 
            self.epoch = epoch
            training_logs = self._train_test_one_epoch()
            log_metrics(epoch, self.args.epochs, training_logs)
            self._eval_()
        self._save_model()
        self.writer.close()
        print("TRAINING ENDED")
    

    def _eval_(self):
        eval_mse = self._eval_model()
        self.min_mse = min(eval_mse, self.min_mse)
        print(f"""
              MIN_MSE = {self.min_mse}
              """)


    def _save_model(self): 
        eval_mse = self._eval_model()
        model_path = os.path.join(self.model_dir, f"model.pth")
        torch.save({
            'model': self.model,
            'eval_mse': eval_mse
        }, model_path)
        print(f'SAVED FINAL MODEL TO DIR ' + self.final_model_dir)
        self._eval_()


    def _eval_model(self):
        running_eval_mse = 0.
        device = self.args.device if torch.cuda.is_available() else "cpu"
        with tqdm(self.dataloader_eval) as iterator:
            for batch in iterator:
            
                imgs, targets, _ = batch

                targets = targets.float()
                imgs, targets = imgs.to(device), targets.to(device)
                _, outputs = self._test_one_batch(imgs=imgs, targets=targets)

                # Calculate and accumulate metrics
                eval_mse = self.loss(outputs, targets)
                running_eval_mse += eval_mse

        return running_eval_mse / len(self.dataloader_eval)
                

    def _init_transforms(self):

        # init training transforms 
        if self.args.augmentations:
            match self.args.crop:
                case 'fixed':
                    scale=(0.9, 0.9)
                case 'random':
                    scale=(0.8, 1)

            self.train_transforms = transforms.Compose([
                transforms.Resize(self.args.img_size),
                transforms.RandomApply(nn.ModuleList([transforms.RandomResizedCrop(self.args.img_size, scale=scale)]), p=self.args.crop_prob),
                RandomHorizontalFlip(p=self.args.hflip_prob),
                RandomVerticalFlip(p=self.args.vflip_prob),
                transforms.Grayscale(self.args.num_channels), 
                transforms.ToTensor()
            ])
        else:
            self.train_transforms = transforms.Compose([
                transforms.Resize(self.args.img_size),
                transforms.Grayscale(self.args.num_channels), 
                transforms.ToTensor()
            ])

        # init test and eval transforms 
        self.test_transforms = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.Grayscale(self.args.num_channels), 
            transforms.ToTensor()
        ])
    

    def _create_dataloaders(self): 

        # get dataset
        dataset_train = SamData.get_sam_dataset(mode=self.args.mode, img_size=self.args.img_size, transforms=self.train_transforms, split='train')
        dataset_test = SamData.get_sam_dataset(mode=self.args.mode, img_size=self.args.img_size, transforms=self.test_transforms, split='test')
        dataset_eval = SamData.get_sam_dataset(mode=self.args.mode, img_size=self.args.img_size, transforms=self.test_transforms, split='eval')

        # create dataloaders 
        self.dataloader_train = DataLoader(dataset_train, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory = self.args.pin_memory)
        self.dataloader_test = DataLoader(dataset_test, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory = self.args.pin_memory)
        self.dataloader_eval = DataLoader(dataset_eval, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory = self.args.pin_memory)


    def _to_device(self): 
        device = self.args.device if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.loss.to(device)
            

    def _train_test_one_epoch(self): 
        device = self.args.device if torch.cuda.is_available() else "cpu"

        running_loss_train = 0.0    
        running_loss_test = 0.0

        print("-------  TRAINING  -------")
        with tqdm(self.dataloader_train) as iterator:
            idx_loss = 0 
            for batch in iterator:
               
                imgs, targets, _ = batch
                targets = targets.float()
                imgs, targets = imgs.to(device), targets.to(device)
                loss, _ = self._train_one_batch(imgs=imgs, 
                                                targets=targets)

                running_loss_train += loss.cpu().detach().item()

                self.writer.add_scalar("Running Training Loss", loss.cpu().detach().item(), idx_loss)
                idx_loss += 1

        print("-------  TESTING  -------")
        with tqdm(self.dataloader_test) as iterator:
            idx_loss = 0 
            for batch in iterator:

                imgs, targets, _ = batch
                targets = targets.float()
                imgs, targets = imgs.to(device), targets.to(device)
                
                loss, _ = self._test_one_batch(imgs=imgs, 
                                               targets=targets)

                running_loss_test += loss.cpu().detach().item()

                self.writer.add_scalar("Running Testing Loss", loss.cpu().detach().item(), idx_loss)
                idx_loss += 1

        # Log epoch-level metrics
        self.writer.add_scalar("Epoch Training Loss", running_loss_train/len(self.dataloader_train), self.epoch)
        self.writer.add_scalar("Epoch Testing Loss", running_loss_test/len(self.dataloader_test), self.epoch)

        return {"train_loss": running_loss_train / len(self.dataloader_train),
                "test_loss": running_loss_test / len(self.dataloader_test)}


    def _train_one_batch(self, imgs=None, targets=None): 

        # Set training mode
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass 
        outputs = self.model(imgs).squeeze()

        # Calculate loss 
        loss = self.loss(outputs, targets)

        loss.backward()  
        self.optimizer.step()  

        return loss, outputs


    @torch.no_grad
    def _test_one_batch(self, imgs=None, targets=None): 

        # Set eval mode
        self.model.eval()

        # Forward pass 
        outputs = self.model(imgs).squeeze()

        # Calculate loss 
        loss = self.loss(outputs, targets) 

        return loss, outputs


def log_metrics(epoch, epochs, logs):
    logging.info(f"EPOCH [{epoch}/{epochs}]")
    logging.info(f"""[train/test]: 
                LOSS = [{logs['train_loss']:.4f}/{logs['test_loss']:.4f}] 
                """)
    