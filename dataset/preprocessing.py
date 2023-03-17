import torch
import torch.nn.functional as F
import math
import numpy as np


def pre_process_data(source_img, target_img, device, mean_vector=[0.485, 0.456, 0.406],
                            std_vector=[0.229, 0.224, 0.225], apply_flip=False):

    # img has shape bx3xhxw
    b, _, h_scale, w_scale = target_img.shape

    # original resolution
    if h_scale < 256:
        int_preprocessed_height = 256
    else:
        int_preprocessed_height = int(math.floor(int(h_scale / 8.0) * 8.0))

    if w_scale < 256:
        int_preprocessed_width = 256
    else:
        int_preprocessed_width = int(math.floor(int(w_scale / 8.0) * 8.0))

    if apply_flip:
        # flip the target image horizontally
        target_img_original = target_img
        target_img = []
        for i in range(b):
            transformed_image = np.fliplr(target_img_original[i].cpu().permute(1, 2, 0).numpy())
            target_img.append(transformed_image)

        target_img = torch.from_numpy(np.uint8(target_img)).permute(0, 3, 1, 2)

    source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                      size=(int_preprocessed_height, int_preprocessed_width),
                                                      mode='area')
    target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                      size=(int_preprocessed_height, int_preprocessed_width),
                                                      mode='area')
    source_img_copy = source_img_copy.div(255.0)
    target_img_copy = target_img_copy.div(255.0)
    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

    # resolution 256x256
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device), size=(256, 256), mode='area')
    target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device), size=(256, 256), mode='area')
    source_img_256 = source_img_256.div(255.0)
    target_img_256 = target_img_256.div(255.0)
    source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])

    ratio_x = float(w_scale) / float(int_preprocessed_width)
    ratio_y = float(h_scale) / float(int_preprocessed_height)
    return source_img_copy.to(device), target_img_copy.to(device), source_img_256.to(device), \
        target_img_256.to(device), ratio_x, ratio_y


def pre_process_image(source_img, device, mean_vector=[0.485, 0.456, 0.406],
                             std_vector=[0.229, 0.224, 0.225]):
    b, _, h_scale, w_scale = source_img.shape
    source_img_copy = source_img.float().to(device).div(255.0)

    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

    # resolution 256x256
    source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                     size=(256, 256),
                                                     mode='area').byte()

    source_img_256 = source_img_256.float().div(255.0)
    source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
    return source_img_copy.to(device), source_img_256.to(device)


class BatchPreprocessing:
    def __init__(self, params, apply_mask=False, apply_mask_zero_borders=False,
                 bath_creator=None, appearance_transform_source=None,
                 appearance_transform_target=None,
                 appearance_transform_source_prime=None):

        self.params = params

        self.apply_mask = apply_mask
        self.apply_mask_zero_borders = apply_mask_zero_borders

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bath_creator = bath_creator

        self.appearance_transform_source = appearance_transform_source
        self.appearance_transform_target = appearance_transform_target
        self.appearance_transform_source_prime = appearance_transform_source_prime

    def __call__(self, batch, net=None, training=False,  *args, **kwargs):

        # create the image triplet from the image pair
        if self.bath_creator is not None:
            batch = self.bath_creator(batch)

        # add appearance augmentation to all three images
        # appearance transform on tensor arrays, which are already B, C, H, W
        if self.appearance_transform_source is not None:
            batch['source_image'] = self.appearance_transform_source(batch['source_image'])

        if self.appearance_transform_target is not None:
            batch['target_image'] = self.appearance_transform_target(batch['target_image'])

        if self.appearance_transform_source_prime is not None:
            batch['source_image_prime1'] = self.appearance_transform_source_prime(batch['source_image_prime1'])
            batch['source_image_prime2'] = self.appearance_transform_source_prime(batch['source_image_prime2'])

        source_image, source_image_256 = pre_process_image(batch['source_image'], self.device)
        target_image, target_image_256 = pre_process_image(batch['target_image'], self.device)
        source_image_prime1, source_image_prime1_256 = pre_process_image(batch['source_image_prime1'],
                                                                              self.device)
        source_image_prime2, source_image_prime2_256 = pre_process_image(batch['source_image_prime2'],
                                                                              self.device)

        flow_gt1 = batch['flow_map1'].to(self.device)
        flow_gt2 = batch['flow_map2'].to(self.device)

        if flow_gt1.shape[1] != 2:
            flow_gt1.permute(0, 3, 1, 2)
        if flow_gt2.shape[1] != 2:
            flow_gt2.permute(0, 3, 1, 2)

        bs, _, h_original, w_original = flow_gt1.shape

        # ground-truth flow field for 256x256 module
        flow_gt1_256 = F.interpolate(flow_gt1, (256, 256), mode='bilinear', align_corners=False)
        flow_gt1_256[:, 0, :, :] *= 256.0 / float(w_original)
        flow_gt1_256[:, 1, :, :] *= 256.0 / float(h_original)

        flow_gt2_256 = F.interpolate(flow_gt2, (256, 256), mode='bilinear', align_corners=False)
        flow_gt2_256[:, 0, :, :] *= 256.0 / float(w_original)
        flow_gt2_256[:, 1, :, :] *= 256.0 / float(h_original)

        mask1 = None
        mask1_256 = None

        mask2 = None
        mask2_256 = None
        if self.apply_mask_zero_borders:
            if 'mask_zero_borders1' not in batch.keys() or 'mask_zero_borders2' not in batch.keys():
                raise ValueError('Mask zero borders not in mini batch')
            mask1 = batch['mask_zero_borders1'].to(self.device)
            mask2 = batch['mask_zero_borders2'].to(self.device)

        if mask1 is not None and (mask1.shape[1] != h_original or mask1.shape[2] != w_original):
            # mask_gt does not have the proper shape
            mask1 = F.interpolate(mask1.float().unsqueeze(1), (h_original, w_original), mode='bilinear',
                                 align_corners=False).squeeze(1).byte()  # bxhxw
            mask1 = mask1.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask1.byte()

        if mask1 is not None:
            mask1_256 = F.interpolate(mask1.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                     align_corners=False).squeeze(1).byte()  # bx256x256, rounding
            mask1_256 = mask1_256.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask1_256.byte()

        if mask2 is not None and (mask2.shape[1] != h_original or mask2.shape[2] != w_original):
            # mask_gt does not have the proper shape
            mask2 = F.interpolate(mask2.float().unsqueeze(1), (h_original, w_original), mode='bilinear',
                                 align_corners=False).squeeze(1).byte()  # bxhxw
            mask2 = mask2.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask2.byte()

        if mask2 is not None:
            mask2_256 = F.interpolate(mask2.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                     align_corners=False).squeeze(1).byte()  # bx256x256, rounding
            mask2_256 = mask2_256.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask2_256.byte()

        batch['source_image'] = source_image
        batch['target_image'] = target_image
        batch['source_image_prime1'] = source_image_prime1
        batch['source_image_prime2'] = source_image_prime2
        batch['source_image_256'] = source_image_256
        batch['target_image_256'] = target_image_256
        batch['source_image_prime1_256'] = source_image_prime1_256
        batch['source_image_prime2_256'] = source_image_prime2_256

        batch['correspondence_mask1'] = batch['correspondence_mask1'].to(self.device)
        batch['correspondence_mask2'] = batch['correspondence_mask2'].to(self.device)
        batch['mask1'] = mask1
        batch['mask2'] = mask2
        batch['flow_map1'] = flow_gt1
        batch['flow_map2'] = flow_gt2
        batch['mask1_256'] = mask1_256
        batch['mask2_256'] = mask2_256
        batch['flow_map1_256'] = flow_gt1_256
        batch['flow_map2_256'] = flow_gt2_256

        return batch