from torchvision import transforms
from torch.utils.data import Dataset
from Configs import config  
from datasets import load_dataset

class ImageNet100(Dataset):
    def __init__(self, img_size, split, transform=None):
        df = load_dataset("clane9/imagenet-100", cache_dir=config.image_net_path)
        self.df = df['train'] if split=='train' else df['validation']
        self.img_size = img_size
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.RandomCrop((160, 160)),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df[idx]['image']
        transformed_image = self.transform(image)
        return transformed_image