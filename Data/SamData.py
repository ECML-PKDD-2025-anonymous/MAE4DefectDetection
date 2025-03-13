import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Configs import config  

""" Here, we implement the dataset classes for all experiments involving SAM images. """

# Returns SAM dataset according to the given mode
def get_sam_dataset(mode, img_size, transforms, split):
    match mode:
        case 'pretrain':
            dataset = SamDataAllImages(images_path=config.sam_crops_directory, img_size=img_size, transform=transforms, split=split)
        case _:
            dataset = SamDataRULRegression(images_path=config.sam_crops_directory, img_size=img_size, transform=transforms, split=split)
    return dataset

# Base Interface for all SAM image datasets 
class SamDataInterface(Dataset):
    def __init__(self, images_path, img_size, transform, split, mask_path=None):
        self.images_path = images_path
        self.mask_path = mask_path
        self.split = split
        self.df = self.get_dataframe()
        self.img_size = img_size
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor()])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        pass

    def get_dataframe(self):
        pass

    def open_image(self, image_path):
        img = Image.open(image_path)
        img_transformed = self.transform(img)

        return img_transformed
    
    def open_mask(self, mask_path):
        mask = Image.open(mask_path) if mask_path is not None else None
        mask_transformed = self.mask_transform(mask) if mask is not None else torch.zeros((1, self.img_size, self.img_size))

        return mask_transformed

# Dataset Class including all available images, 
# as required for pretraining
class SamDataAllImages(SamDataInterface):

    def __init__(self, images_path, img_size, transform=None, split=None):
        super().__init__(images_path, img_size, transform, split)


    def get_dataframe(self):
        df = get_dataframe_for_steps(self.images_path, 0, split=self.split)
        return df

    def __getitem__(self, idx):
        data_point = self.df.iloc[idx]

        image = self.open_image(data_point[0])

        return image

class SamDataRULRegression(SamDataInterface):
    def __init__(self, images_path, img_size, final_tta_path=None, transform=None, split=None):
        super().__init__(images_path, img_size, transform, split)

    def get_dataframe(self):
         
        df = get_dataframe_for_tta(images_path=config.sam_crops_directory, 
                                    tta_path=config.tta_pkl_path, 
                                    split=self.split,
                                    test_flag=False)
        return df 

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # Get image 
        image = self.open_image(item[0])

        # Get remaining useful lifetime
        tsc = get_tsc_from_timestamp(item[0])
        tta_complete = item[1]
        
        match tsc:
            case 0:
                tta_index = 0
            case 100:
                tta_index = 3
            case 500: 
                tta_index = 7
            case 1000: 
                tta_index = 9
        if len(tta_complete) > tta_index: 
            tta_for_tsc = tta_complete[tta_index]
        else:
            tta_for_tsc = item[2]
        
        # Get BMAX at 1250 TSC
        bmax1250 = item[2]
        
        return image, tta_for_tsc/tta_complete[0] - 1, bmax1250


# Implements search logic in the SAM image folder structure to return the 
# required image paths along with the associated LED type 
def get_dataframe_for_tta(images_path, tta_path, split, test_flag=False):
    print(f'Creating Dataframe for {split} split.')
    data_frame = []
    img_not_found_counter = 0
    dir_not_found_counter = 0
    img_found_counter = 0
    dir_found_counter = 0
    timestamps = ["0000TSC", "0100TSC", "0500TSC", "1000TSC"] # Dont use 1500 tsc since we predict BMAX at 1250 TSC
    tta_dataframe = pd.read_pickle(tta_path)

    # Use 60% of images for training, 20% for testing and 20% for evaluation 
    match split:
        case 'all': 
            led_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        case 'train':
            led_numbers = [1, 2, 3, 4, 5, 6]
        case 'test':
            led_numbers = [7, 8]
        case 'eval':
            led_numbers = [9, 10]
        case _:
            raise ValueError(f'Split {split} not defined.')

    for panel_type in config.panel_types:
        for panel_nr in range(1, 5):
            for led_type in config.led_types:
                for led_nr in led_numbers:
                    image_time_series = []
                    for timestamp in timestamps:
                        
                        # Get image from directory 
                        directory_string = find_directory(images_path, panel_type, panel_nr, timestamp, test_flag)
                        image_path = find_img_in_directory(directory_string, led_type, led_nr, test_flag)
                        img = os.path.join(directory_string, image_path) if directory_string is not None and image_path is not None else None

                        if test_flag:
                            if directory_string is None:
                                dir_not_found_counter += 1
                            else:
                                dir_found_counter += 1
                            if image_path is None:
                                img_not_found_counter += 1
                            else:
                                img_found_counter += 1
                        
                        if img is not None and not ('1500tsc' in img.lower()): 
                            image_time_series.append(img)

                    # Get corresponding lifetime expectancy for LED from dataframe
                    for index in range(len(tta_dataframe)):
                        elem = tta_dataframe.iloc[index]
                        if elem['key'].lower() == led_type.lower() + '_' + str(led_nr) + '_' + panel_type.lower() + '_' + str(panel_nr):
                            tta = elem['tta']
                            bmax_1250 = elem['bmax_1250']
                            break
                    tuples = get_tuples_all(image_time_series)

                    for data_point in tuples:
                        data_frame.append([data_point, tta, bmax_1250])

    pandas_data_frame = pd.DataFrame(data_frame)

    if test_flag:
        print("Total images available:" + str(count_images_in_directory(config.sam_crops_directory)))
        print("sum images not found: " + str(img_not_found_counter))
        print("sum directories not found: " + str(dir_not_found_counter))
        print("sum images found: " + str(img_found_counter))
        print("sum directories found: " + str(dir_found_counter))

    return pandas_data_frame


# Implements search logic in the SAM image folder structure to return the 
# required image paths for sets of images of the same LED 
def get_dataframe_for_steps(images_path, steps, split, test_flag=False):
    data_frame = []
    img_not_found_counter = 0
    dir_not_found_counter = 0
    img_found_counter = 0
    dir_found_counter = 0
    timestamps = ["0000TSC", "0100TSC", "0500TSC", "1000TSC", "1500TSC"]

    # TSC0100 is only used in pretraining, as we always work with 500TSC steps in downstream tasks
    if steps == 1 or steps == 2:
        timestamps.remove('0100TSC')

    match split:
        case 'train':
            led_numbers = [1, 2, 3, 4, 5, 6]
        case 'test':
            led_numbers = [7, 8]
        case 'eval':
            led_numbers = [9, 10]
        case _:
            raise ValueError(f'Split {split} not defined.')

    for panel_type in config.panel_types:
        for panel_nr in range(1, 5):
            for led_type in config.led_types:
                for led_nr in led_numbers:
                    image_time_series = []
                    for timestamp in timestamps:

                        directory_string = find_directory(images_path, panel_type, panel_nr, timestamp, test_flag)
                        image_path = find_img_in_directory(directory_string, led_type, led_nr, test_flag)
                        img = os.path.join(directory_string, image_path) if directory_string is not None and image_path is not None else None

                        if test_flag:
                            if directory_string is None:
                                dir_not_found_counter += 1
                            else:
                                dir_found_counter += 1
                            if image_path is None:
                                img_not_found_counter += 1
                            else:
                                img_found_counter += 1

                        image_time_series.append(img)

                    match steps:
                        case 0:
                            tuples = get_tuples_all(image_time_series)
                        case 1:
                            tuples = get_tuples_one_step(image_time_series)
                        case 2:
                            tuples = get_tuples_two_step(image_time_series)
                        case _:
                            raise ValueError(f"Invalid step count ({steps}) when creating dataframe.")

                    for data_point in tuples:
                        data_frame.append(data_point)

    pandas_data_frame = pd.DataFrame(data_frame)

    if test_flag:
        print("Total images available:" + str(count_images_in_directory(config.sam_crops_directory)))
        print("sum images not found: " + str(img_not_found_counter))
        print("sum directories not found: " + str(dir_not_found_counter))
        print("sum images found: " + str(img_found_counter))
        print("sum directories found: " + str(dir_found_counter))

    return pandas_data_frame

def get_dataframe_for_timeseries(images_path, split, test_flag=False):
    data_frame = []
    img_not_found_counter = 0
    dir_not_found_counter = 0
    img_found_counter = 0
    dir_found_counter = 0
    timestamps = ["0000TSC", "0100TSC", "0500TSC", "1000TSC", "1500TSC"]

    match split:
        case 'train':
            led_numbers = [1, 2, 3, 4, 5, 6]
        case 'test':
            led_numbers = [7, 8]
        case 'eval':
            led_numbers = [9, 10]
        case _:
            raise ValueError(f'Split {split} not defined.')

    for panel_type in config.panel_types:
        for panel_nr in range(1, 5):
            for led_type in config.led_types:
                for led_nr in led_numbers:
                    image_time_series = []
                    for timestamp in timestamps:

                        directory_string = find_directory(images_path, panel_type, panel_nr, timestamp, test_flag)
                        image_path = find_img_in_directory(directory_string, led_type, led_nr, test_flag)
                        img = os.path.join(directory_string, image_path) if directory_string is not None and image_path is not None else None

                        if test_flag:
                            if directory_string is None:
                                dir_not_found_counter += 1
                            else:
                                dir_found_counter += 1
                            if image_path is None:
                                img_not_found_counter += 1
                            else:
                                img_found_counter += 1

                        image_time_series.append(img)

                    
                        tuples = get_tuples_all(image_time_series)
                        data_frame.append(tuples)


    return data_frame

def one_hot_encode_type(type_str):
    vector = torch.zeros([9])
    led_types = config.led_types

    for index in range(len(led_types)):
        if led_types[index].lower() == type_str.lower():
            vector[index] = 1
            
    return vector

def get_tuples_all(time_series):
    valid_tuples = []
    for image in range(len(time_series)):
        if time_series[image] is not None:
            valid_tuples.append(time_series[image])
    return valid_tuples

def get_LED_type(image_path):
    type = None
    path_lower = image_path.lower()
    for LED_type in config.led_types: 
        if LED_type.lower() in path_lower:
            type = LED_type
            break
    if type is None: 
        raise Exception(f'Did not find a LED type for given image: {image_path}')
    return type

def get_tsc_from_timestamp(path):
    if '0000tsc' in path.lower(): 
        return 0
    elif '0100tsc' in path.lower():
        return 100
    elif '0500tsc' in path.lower():
        return 500
    elif '1000tsc' in path.lower():
        return 1000
    
    raise ValueError(f'TSC not found in string: {path}')

def find_img_in_directory(directory, led_type, led_nr, test_flag, segmentation_flag=None):
    # Logic for the segmentation case 
    if segmentation_flag is not None:
        if segmentation_flag.lower() not in ['void', 'crack']: 
            raise ValueError(f'Segmentation flag {segmentation_flag} is not a valid flag. Use void or crack as flags.')
        for image in os.listdir(directory):
            if (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led_" + str(led_nr) + "_" in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led_" + str(led_nr) + "." in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led_" + str(led_nr) + "_" in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led_0" + str(led_nr) + "." in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led_0" + str(led_nr) + "_" in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led" + str(led_nr) + "." in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led" + str(led_nr) + "_" in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led0" + str(led_nr) + "." in image.lower()) \
                    or (segmentation_flag.lower() in image.lower() and led_type.lower() in image.lower() and "led0" + str(led_nr) + "_" in image.lower()):
                image_path = image
                return image_path
                        
    # Logic for all other cases         
    else: 
        for image in os.listdir(directory):
            if (led_type.lower() in image.lower() and "led_" + str(led_nr) + "_" in image.lower()) \
                    or (led_type.lower() in image.lower() and "led_" + str(led_nr) + "." in image.lower()) \
                    or (led_type.lower() in image.lower() and "led_" + str(led_nr) + "_" in image.lower()) \
                    or (led_type.lower() in image.lower() and "led_0" + str(led_nr) + "." in image.lower()) \
                    or (led_type.lower() in image.lower() and "led_0" + str(led_nr) + "_" in image.lower()) \
                    or (led_type.lower() in image.lower() and "led" + str(led_nr) + "." in image.lower()) \
                    or (led_type.lower() in image.lower() and "led" + str(led_nr) + "_" in image.lower()) \
                    or (led_type.lower() in image.lower() and "led0" + str(led_nr) + "." in image.lower()) \
                    or (led_type.lower() in image.lower() and "led0" + str(led_nr) + "_" in image.lower()):
                image_path = image
                return image_path
    if test_flag:
        print(f"Image not found for LED type {led_type}, LED number {led_nr}")
    return None


def find_directory(image_dir, panel_type, panel_nr, timestamp, test_flag):
    for directory in os.listdir(os.path.join(image_dir, timestamp)):
        if panel_type.lower() in directory.lower() and ('panel' + str(
                panel_nr)).lower() in directory.lower() and timestamp.lower() in directory.lower():
            return os.path.join(image_dir, timestamp, directory)
    if test_flag:
        print(f"Directory not found for panel name {panel_type}, "
              f"panel number {str(panel_nr)}, timestamp {timestamp}.")
    return None

def count_images_in_directory(directory):
    counter = 0
    for timestamp_dir in os.listdir(directory):
        for panel_dir in os.listdir(os.path.join(directory, timestamp_dir)):
            for _ in os.listdir(os.path.join(directory, timestamp_dir, panel_dir)):
                counter += 1
    return counter
