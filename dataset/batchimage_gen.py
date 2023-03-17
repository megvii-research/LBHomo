import torch
from utils_operations.pixel_wise_mapping import warp
from utils_operations.flow_and_mapping_operations import create_border_mask
import random
import numpy as np


def define_mask_zero_borders(image, epsilon=1e-6):
    """Computes the binary mask, equal to 0 when image is 0 and 1 otherwise."""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            if image.shape[1] == 3:
                # image b, 3, H, W
                image = image.transpose(0, 2, 3, 1)
            # image is b, H, W, 3
            occ_mask = np.logical_and(np.logical_and(image[:, :, :, 0] < epsilon,
                                                     image[:, :, :, 1] < epsilon),
                                      image[:, :, :, 2] < epsilon)
        else:
            if image.shape[0] == 3:
                # image 3, H, W
                image = image.transpose(1, 2, 0)
            # image is H, W, 3
            occ_mask = np.logical_and(np.logical_and(image[:, :, 0] < epsilon,
                                                     image[:, :, 1] < epsilon),
                                      image[:, :, 2] < epsilon)
        mask = ~occ_mask
        mask = mask.astype(np.bool) if float(torch.__version__.split('.')[1]) >= 1 else mask.astype(np.uint8)
    else:
        # torch tensor
        if len(image.shape) == 4:
            if image.shape[1] == 3:
                # image b, 3, H, W
                image = image.permute(0, 2, 3, 1)
            occ_mask = image[:, :, :, 0].le(epsilon) & image[:, :, :, 1].le(epsilon) & image[:, :, :, 2].le(epsilon)
        else:
            if image.shape[0] == 3:
                # image 3, H, W
                image = image.permute(1, 2, 0)
            occ_mask = image[:, :, 0].le(epsilon) & image[:, :, 1].le(epsilon) & image[:, :, 2].le(epsilon)
        mask = ~occ_mask
        mask = mask.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask.byte()
    return mask


class BatchedImageCreation:
    def __init__(self, params, synthetic_flow_generator, compute_mask_zero_borders=False,
                 min_percent_valid_corr=0.1, padding_mode='zeros'):

        self.params = params

        self.compute_mask_zero_borders = compute_mask_zero_borders

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.synthetic_flow_generator = synthetic_flow_generator

        self.crop_size = self.params.crop_size

        self.min_percent_valid_corr = min_percent_valid_corr
        self.padding_mode = padding_mode

    def __call__(self, batch, *args, **kwargs):

        # take original images

        ori_images = batch['ori_images']

        source_image = ori_images[:, :3, :, :].to(self.device)
        target_image = ori_images[:, 3:, :, :].to(self.device)

        b, _, h, w = source_image.shape

        flow_gt1 = self.synthetic_flow_generator(img_size=(h, w),
                                                     random_hom=self.params.random_homo).to(self.device)

        flow_gt2 = self.synthetic_flow_generator(img_size=(h, w), random_hom=self.params.random_homo).to(self.device)
        flow_gt2 = warp(flow_gt2, flow_gt1) + flow_gt1

        flow_gt2.require_grad = False

        source_image_prime1 = warp(source_image, flow_gt1, padding_mode=self.padding_mode).byte()
        source_image_prime2 = warp(source_image, flow_gt2, padding_mode=self.padding_mode).byte()

        # crop a center patch from the images and the ground-truth flow field, so that black borders are removed
        x_start = random.randint(self.params.rho, w-self.params.rho-self.crop_size[1])  # w // 2 - self.crop_size[1] // 2
        y_start = random.randint(self.params.rho, h-self.params.rho-self.crop_size[0])  # h // 2 - self.crop_size[0] // 2

        source_image_resized = source_image[:, :, y_start: y_start + self.crop_size[0],
                               x_start: x_start + self.crop_size[1]]
        target_image_resized = target_image[:, :, y_start: y_start + self.crop_size[0],
                               x_start: x_start + self.crop_size[1]]

        source_image_prime1_resized = source_image_prime1[:, :, y_start: y_start + self.crop_size[0],
                                     x_start: x_start + self.crop_size[1]]
        source_image_prime2_resized = source_image_prime2[:, :, y_start: y_start + self.crop_size[0],
                                     x_start: x_start + self.crop_size[1]]

        flow_gt1_resized = flow_gt1[:, :, y_start: y_start + self.crop_size[0],
                          x_start: x_start + self.crop_size[1]]
        flow_gt2_resized = flow_gt2[:, :, y_start: y_start + self.crop_size[0],
                          x_start: x_start + self.crop_size[1]]

        # create ground truth correspondence mask for flow between target prime and target
        mask_gt1 = create_border_mask(flow_gt1_resized)
        mask_gt1 = mask_gt1.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask_gt1.byte()

        mask_gt2 = create_border_mask(flow_gt2_resized)
        mask_gt2 = mask_gt2.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask_gt2.byte()

        if self.compute_mask_zero_borders:
            # if mask_gt has too little commun areas, overwrite to use that mask in anycase
            if mask_gt1.sum() < mask_gt1.shape[-1] * mask_gt1.shape[-2] * self.min_percent_valid_corr:
                mask1 = mask_gt1
            else:
                # mask black borders that might have appeared from the warping, when creating target_image_prime
                mask1 = define_mask_zero_borders(source_image_prime1_resized)
            batch['mask_zero_borders1'] = mask1

            if mask_gt2.sum() < mask_gt2.shape[-1] * mask_gt2.shape[-2] * self.min_percent_valid_corr:
                mask2 = mask_gt2
            else:
                # mask black borders that might have appeared from the warping, when creating target_image_prime
                mask2 = define_mask_zero_borders(source_image_prime2_resized)
            batch['mask_zero_borders2'] = mask2

        # save the new batch information
        batch['source_image'] = source_image_resized.byte()
        batch['source_image_prime1'] = source_image_prime1_resized.byte()
        batch['source_image_prime2'] = source_image_prime2_resized.byte()
        batch['target_image'] = target_image_resized.byte()
        batch['correspondence_mask1'] = mask_gt1
        batch['correspondence_mask2'] = mask_gt2
        batch['flow_map1'] = flow_gt1_resized
        batch['flow_map2'] = flow_gt2_resized
        return batch