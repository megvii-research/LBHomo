import logging
import os
import pickle
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


_logger = logging.getLogger(__name__)


class HomoTrainData(Dataset):

    def __init__(self, params):

        self.horizontal_flip_aug = True
        self.position_change = False

        self.data_list = os.path.join(params.data_dir, "large_train_1.txt")
        self.image_path = os.path.join(params.data_dir, "img")
        self.data_infor = open(self.data_list, 'r').readlines()

        print(self.data_list)

        self.seed = 0
        random.seed(self.seed)
        random.shuffle(self.data_infor)

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):
        # img loading
        img_pair = self.data_infor[idx].replace('\n', '')
        pari_id = img_pair.split(' ')
        video_name = pari_id[0].split('_')[0]

        img_names = img_pair.replace('LM', '').replace(video_name + '/', '').split(' ')
        img_files = os.path.join(self.image_path, video_name)

        img1 = cv2.imread(os.path.join(img_files, img_names[0] + '.jpg'))
        img2 = cv2.imread(os.path.join(img_files, img_names[1] + '.jpg'))

        if self.position_change and random.random() <= .5:
            img1, img2 = img2, img1

        if self.horizontal_flip_aug and random.random() <= .5:
            img1 = np.flip(img1, 1)
            img2 = np.flip(img2, 1)

        # array to tensor
        ori_images = torch.tensor(np.concatenate([img1, img2], axis=2)).permute(2, 0, 1).float()

        data_dict = {'ori_images': ori_images}

        return data_dict


class HomoTestData(Dataset):

    def __init__(self, params):

        self.data_list = os.path.join(params.data_dir, "large_test.txt")
        self.npy_path = os.path.join(params.data_dir, "npy")
        self.image_path = os.path.join(params.data_dir, "img")

        self.data_infor = open(self.data_list, 'r').readlines()

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):

        # img loading
        img_pair = self.data_infor[idx].replace('\n', '')
        pari_id = img_pair.split(' ')
        npy_name = pari_id[0] + '_' + pari_id[1] + '.npy'
        video_name = pari_id[0].split('_')[0]

        img_names = img_pair.replace('LM', '').replace(video_name + '/', '').split(' ')
        img_files = os.path.join(self.image_path, video_name)

        img1 = cv2.imread(os.path.join(img_files, img_names[0] + '.jpg'))
        img2 = cv2.imread(os.path.join(img_files, img_names[1] + '.jpg'))

        # array to tensor
        ori_images = torch.tensor(np.concatenate([img1, img2], axis=2)).permute(2, 0, 1).float()

        points_path = os.path.join(self.npy_path, npy_name)

        # output dict
        data_dict = {"ori_images": ori_images, "points_path": points_path,
                     "video_name": video_name, "npy_name": npy_name}

        return data_dict


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    _logger.info("Dataset type: {}, transform type: {}".format(params.dataset_type, params.transform_type))

    if params.dataset_type == "basic":
        train_ds = HomoTrainData(params)
        test_ds = HomoTestData(params)

    dataloaders = {}
    # add train data loader
    train_dl = DataLoader(
        train_ds,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
        drop_last=True,
        # prefetch_factor=3, # for pytorch >=1.5.0
    )
    dataloaders["train"] = train_dl

    # chose test data loader for evaluate

    if params.eval_type == "test":
        dl = DataLoader(
            test_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.cuda
            # prefetch_factor=3, # for pytorch >=1.5.0
        )
    else:
        dl = None
        raise ValueError("Unknown eval_type in params, should in [val, test]")

    dataloaders[params.eval_type] = dl

    return dataloaders
