import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from .losses.basic_losses import L1
from .losses.flow_identity_loss import IdentityLoss
from .losses.multiscale_loss import MultiScaleFlow
from .DLT import homo_flow_gen
from .error_compute import compute_error, identity_error
from dataset.preprocessing import BatchPreprocessing, pre_process_data
from dataset.homo_sampling import Synthetic_Homo, Synthetic_HomoFlow
from dataset.batchimage_gen import BatchedImageCreation
from dataset.color_aug import ColorJitter, RandomGaussianBlur


def train_bath_generation(train_batch, params):

    source_image = train_batch['ori_images'][:, :3, :, :]
    _, _, h, w = source_image.shape

    homo_sampling_module = Synthetic_Homo(img_size=(h, w), random_hom=params.random_homo)
    synthetic_flow_generator = Synthetic_HomoFlow(img_size=(h, w), sampling_module=homo_sampling_module)

    cycle_creator = BatchedImageCreation(params, synthetic_flow_generator=synthetic_flow_generator,
                                                  compute_mask_zero_borders=True)

    appearance_transform_source_prime = transforms.Compose([ColorJitter(brightness=0.6, contrast=0.6,
                                                                        saturation=0.6, hue=0.5 / 3.14),
                                                            RandomGaussianBlur(sigma=(0.2, 2.0),
                                                                               probability=0.2)])

    batch_images = BatchPreprocessing(params, apply_mask=False, apply_mask_zero_borders=True,
                                      bath_creator=cycle_creator,
                                      appearance_transform_source=None,
                                      appearance_transform_target=None,
                                      appearance_transform_source_prime=None)

    train_batch_ = batch_images(train_batch)

    return train_batch_


def compute_losses(train_batch, model, params):
    losses = {}

    train_batch = train_bath_generation(train_batch, params)

    objective = L1()

    loss_weight = {'warp_supervision': 1.0, 'w_bipath': 1.0,
                   'warp_supervision_constant': 1.0, 'w_bipath_constant': 1.0,
                   'cc_mask_alpha_1': 0.03, 'cc_mask_alpha_2': 0.5}

    criterion_256 = MultiScaleFlow(level_weights=params.weight_fil[:2], loss_function=objective,
                                   downsample_gt_flow=True)

    criterion = MultiScaleFlow(level_weights=params.weight_fil[2:], loss_function=objective,
                               downsample_gt_flow=True)

    unsupervised_criterion = IdentityLoss(
        criterion, loss_weight, detach_flow_for_warping=True,
        compute_cyclic_consistency=True,
        alpha_1=loss_weight['cc_mask_alpha_1'], alpha_2=loss_weight['cc_mask_alpha_2'])
    unsupervised_criterion_256 = IdentityLoss(
        criterion_256, loss_weight, detach_flow_for_warping=True,
        compute_cyclic_consistency=True,
        alpha_1=loss_weight['cc_mask_alpha_1'], alpha_2=loss_weight['cc_mask_alpha_2'])

    b, _, h, w = train_batch['flow_map1'].shape
    b, _, h_256, w_256 = train_batch['flow_map1_256'].shape

    # ======source_prime1 to source==============
    output_source_prime1_to_source_directly_256, output_source_prime1_to_source_directly = \
        model(train_batch['source_image_prime1'], train_batch['source_image'],
              train_batch['source_image_prime1_256'], train_batch['source_image_256'])
    estimated_flow_source_prime1_to_source_256 = output_source_prime1_to_source_directly_256['flow_estimates']
    estimated_flow_source_prime1_to_source = output_source_prime1_to_source_directly['flow_estimates']

    # ======source_prime2 to source==============
    output_source_prime2_to_source_directly_256, output_source_prime2_to_source_directly = \
        model(train_batch['source_image_prime2'], train_batch['source_image'],
              train_batch['source_image_prime2_256'], train_batch['source_image_256'])
    estimated_flow_source_prime2_to_source_256 = output_source_prime2_to_source_directly_256['flow_estimates']
    estimated_flow_source_prime2_to_source = output_source_prime2_to_source_directly['flow_estimates']

    # ======target to source==============
    output_target_to_source_256, output_target_to_source = \
        model(train_batch['target_image'], train_batch['source_image'],
              train_batch['target_image_256'], train_batch['source_image_256'])
    estimated_flow_target_to_source_256 = output_target_to_source_256['flow_estimates']
    estimated_flow_target_to_source = output_target_to_source['flow_estimates']

    # ======source_prime1 to source==============
    output_source_prime1_to_target_256, output_source_prime1_to_target = \
        model(train_batch['source_image_prime1'], train_batch['target_image'],
              train_batch['source_image_prime1_256'], train_batch['target_image_256'])
    estimated_flow_source_prime1_to_target_256 = output_source_prime1_to_target_256['flow_estimates']
    estimated_flow_source_prime1_to_target = output_source_prime1_to_target['flow_estimates']

    # ======source_prime2 to source==============
    output_source_prime2_to_target_256, output_source_prime2_to_target = \
        model(train_batch['source_image_prime2'], train_batch['target_image'],
              train_batch['source_image_prime2_256'], train_batch['target_image_256'])
    estimated_flow_source_prime2_to_target_256 = output_source_prime2_to_target_256['flow_estimates']
    estimated_flow_source_prime2_to_target = output_source_prime2_to_target['flow_estimates']

    # ================================supervise loss===============================================

    ss_loss_1 = criterion(estimated_flow_source_prime1_to_source,
                                      train_batch['flow_map1'], mask=train_batch['mask1'])
    ss_loss_1_256 = criterion_256(estimated_flow_source_prime1_to_source_256,
                                                       train_batch['flow_map1_256'], mask=train_batch['mask1_256'])

    ss_loss_2 = criterion(estimated_flow_source_prime2_to_source,
                                      train_batch['flow_map2'], mask=train_batch['mask2'])
    ss_loss_2_256 = criterion_256(estimated_flow_source_prime2_to_source_256,
                                                       train_batch['flow_map2_256'], mask=train_batch['mask2_256'])

    ss_loss_o = ss_loss_1 + 0.1 * ss_loss_2
    ss_loss_256 = ss_loss_1_256 + 0.1 * ss_loss_2_256

    losses['supervise'] = ss_loss_o + ss_loss_256

    # ================================unsupervise loss===============================================

    un_loss_1 = unsupervised_criterion(
        train_batch['flow_map1'], train_batch['mask1'],
        estimated_flow_source_prime1_to_target, estimated_flow_target_to_source)

    un_loss_1_256 = unsupervised_criterion_256(
        train_batch['flow_map1_256'], train_batch['mask1_256'],
        estimated_flow_source_prime1_to_target_256, estimated_flow_target_to_source_256)

    un_loss_2 = unsupervised_criterion(
        train_batch['flow_map2'], train_batch['mask2'],
        estimated_flow_source_prime2_to_target, estimated_flow_target_to_source)

    un_loss_2_256 = unsupervised_criterion_256(
        train_batch['flow_map2_256'], train_batch['mask2_256'],
        estimated_flow_source_prime2_to_target_256, estimated_flow_target_to_source_256)

    un_loss_o = un_loss_1 + 0.1 * un_loss_2
    un_loss_256 = un_loss_1_256 + 0.1 * un_loss_2_256

    losses['unsupervise'] = un_loss_o + un_loss_256

    if params.loss_type == 'supervise':
        losses["total"] = losses["supervise"]
    elif params.loss_type == 'unsupervise':
        losses["total"] = losses["unsupervise"]
    else:
        L_unsupervised = losses["unsupervise"].detach()
        L_supervised = losses["supervise"].detach()
        if L_unsupervised > L_supervised:
            u_l_w = 1
            s_l_w = L_unsupervised / (L_supervised + 1e-8)
        else:
            u_l_w = L_supervised / (L_unsupervised + 1e-8)
            s_l_w = 1
        losses["unsupervise"] = losses["unsupervise"] * u_l_w
        losses["supervise"] = losses["supervise"] * s_l_w
        losses["total"] = losses["supervise"] + losses["unsupervise"]

    return losses


def process_test_data(source_img, target_img, apply_flip=False):
    return pre_process_data(source_img, target_img, source_img.device, apply_flip=apply_flip)


def estimate_flow(source_img, target_img, model, output_shape=None,
                  scaling=1.0, mode='channel_first'):

    w_scale = target_img.shape[3]
    h_scale = target_img.shape[2]
    # define output_shape
    if output_shape is None and scaling != 1.0:
        output_shape = (int(h_scale*scaling), int(w_scale*scaling))

    source_img, target_img, source_img_256, target_img_256, ratio_x, ratio_y = \
        process_test_data(source_img, target_img)
    output_256, output = model(target_img, source_img, target_img_256, source_img_256)

    flow_est_list = output['flow_estimates']
    flow_est = flow_est_list[-1]

    if output_shape is not None:
        ratio_x *= float(output_shape[1]) / float(w_scale)
        ratio_y *= float(output_shape[0]) / float(h_scale)
    else:
        output_shape = (h_scale, w_scale)
    flow_est = torch.nn.functional.interpolate(input=flow_est, size=output_shape,
                                               mode='bilinear', align_corners=False)

    flow_est[:, 0, :, :] *= ratio_x
    flow_est[:, 1, :, :] *= ratio_y

    if mode == 'channel_first':
        return flow_est
    else:
        return flow_est.permute(0, 2, 3, 1)


def test_model_on_image_pair(source_image, target_image, network):
    network.eval()
    with torch.no_grad():

        estimated_flow = estimate_flow(source_image, target_image, network, mode='channel_first')

        estimated_flow = homo_flow_gen(estimated_flow)

        estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
        # estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
        return estimated_flow_numpy

