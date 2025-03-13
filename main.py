import hydra
from torch.utils.tensorboard import SummaryWriter
from Training import Train
from datetime import datetime
import os

@hydra.main(version_base=None, config_path="Configs", config_name="config")
def main(cfg) -> None: 

    trainer = Train.Trainer(cfg)

    print("NUMBER OF PARAMETERS FOR THE MODEL ARE: ", trainer.parameters)
    print("DEVICE TRAINED ON: ", next(trainer.model.parameters()).device)
    
    trainer.train()
    

if __name__ == '__main__': 
    main()
