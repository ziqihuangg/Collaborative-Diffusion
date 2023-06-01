import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json
import random
from ldm.util import instantiate_from_config_vq_diffusion
import albumentations
from torchvision import transforms as trans

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DalleTransformerPreprocessor(object):
    def __init__(self,
                 size=256,
                 phase='train',
                 additional_targets=None):

        self.size = size
        self.phase = phase
        # ddc: following dalle to use randomcrop
        self.train_preprocessor = albumentations.Compose([albumentations.RandomCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)
        self.val_preprocessor = albumentations.Compose([albumentations.CenterCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)


    def __call__(self, image, **kargs):
        """
        image: PIL.Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        w, h = image.size
        s_min = min(h, w)

        if self.phase == 'train':
            off_h = int(random.uniform(3*(h-s_min)//8, max(3*(h-s_min)//8+1, 5*(h-s_min)//8)))
            off_w = int(random.uniform(3*(w-s_min)//8, max(3*(w-s_min)//8+1, 5*(w-s_min)//8)))

            image = image.crop((off_w, off_h, off_w + s_min, off_h + s_min))

            # resize image
            t_max = min(s_min, round(9/8*self.size))
            t_max = max(t_max, self.size)
            t = int(random.uniform(self.size, t_max+1))
            image = image.resize((t, t))
            image = np.array(image).astype(np.uint8)
            image = self.train_preprocessor(image=image)
        else:
            if w < h:
                w_ = self.size
                h_ = int(h * w_/w)
            else:
                h_ = self.size
                w_ = int(w * h_/h)
            image = image.resize((w_, h_))
            image = np.array(image).astype(np.uint8)
            image = self.val_preprocessor(image=image)
        return image



class CelebAConditionalDataset(Dataset):

    """
    This Dataset can be used for:
    - image-only: setting 'conditions' = []
    - image and multi-modal 'conditions': setting conditions as the list of modalities you need

    To toggle between 256 and 512 image resolution, simply change the 'image_folder'
    """

    def __init__(self,
        phase = 'train',
        im_preprocessor_config=None,
        test_dataset_size=3000,
        conditions = ['seg_mask', 'text', 'sketch'],
        image_folder = 'datasets/image/image_512_downsampled_from_hq_1024',
        text_file = 'datasets/text/captions_hq_beard_and_age_2022-08-19.json',
        mask_folder = 'datasets/mask/CelebAMask-HQ-mask-color-palette_32_nearest_downsampled_from_hq_512_one_hot_2d_tensor',
        sketch_folder = 'datasets/sketch/sketch_1x1024_tensor',
        ):

        self.transform = instantiate_from_config_vq_diffusion(im_preprocessor_config)
        self.conditions = conditions
        print(f'conditions = {conditions}')

        self.image_folder = image_folder
        print(f'self.image_folder = {self.image_folder}')

        # conditions directory
        self.text_file = text_file
        print(f'self.text_file = {self.text_file}')
        print(f'start loading text')
        with open(self.text_file, 'r') as f:
            self.text_file_content = json.load(f)
        print(f'end loading text')
        if 'seg_mask' in self.conditions:
            self.mask_folder = mask_folder
            print(f'self.mask_folder = {self.mask_folder}')
        if 'sketch' in self.conditions:
            self.sketch_folder = sketch_folder
            print(f'self.sketch_folder = {self.sketch_folder}')

        # list of valid image names & train test split
        self.image_name_list = list(self.text_file_content.keys())

        # train test split
        if phase == 'train':
            self.image_name_list = self.image_name_list[:-test_dataset_size]
        elif phase == 'test':
            self.image_name_list = self.image_name_list[-test_dataset_size:]
        else:
            raise NotImplementedError
        self.num = len(self.image_name_list)

        # verbose
        print(f'phase = {phase}')
        print(f'number of samples = {self.num}')
        print(f'self.image_name_list[:10] = {self.image_name_list[:10]}')
        print(f'self.image_name_list[-10:] = {self.image_name_list[-10:]}\n')


    def __len__(self):
        return self.num

    def __getitem__(self, index):

        # ---------- (1) get image ----------
        image_name = self.image_name_list[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        image = image.astype(np.float32)/127.5 - 1.0

        # record into data entry
        if len(self.conditions) == 1:
            data = {
                'image': image,
            }
        else:
            data = {
                'image': image,
                'conditions': {}
            }

        # ---------- (2) get text ----------
        if 'text' in self.conditions:
            text = self.text_file_content[image_name]["Beard_and_Age"].lower()
            # record into data entry
            if len(self.conditions) == 1:
                data['caption'] = text
            else:
                data['conditions']['text'] = text

        # ---------- (3) get mask ----------
        if 'seg_mask' in self.conditions:
            mask_idx = image_name.split('.')[0]
            mask_name = f'{mask_idx}.pt'
            mask_path = os.path.join(self.mask_folder, mask_name)
            mask_one_hot_tensor = torch.load(mask_path)

            # record into data entry
            if len(self.conditions) == 1:
                data['seg_mask'] = mask_one_hot_tensor
            else:
                data['conditions']['seg_mask'] = mask_one_hot_tensor


        # ---------- (4) get sketch ----------
        if 'sketch' in self.conditions:
            sketch_idx = image_name.split('.')[0]
            sketch_name = f'{sketch_idx}.pt'
            sketch_path = os.path.join(self.sketch_folder, sketch_name)
            sketch_one_hot_tensor = torch.load(sketch_path)

            # record into data entry
            if len(self.conditions) == 1:
                data['sketch'] = sketch_one_hot_tensor
            else:
                data['conditions']['sketch'] = sketch_one_hot_tensor


        return data
