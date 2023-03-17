import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_operations.model_constructor import model_constructor
from utils_operations.flow_and_mapping_operations import unormalise_flow_or_mapping
from .base_Net import MultiScaleNet
from .base_Net import set_parameters
from ..modules.mod import FlowEstimatorResidualConnection, \
    last_level_refinement_module, adaptive_reso_refinement_module
from ..modules.mod import deconv


class feature_base_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(feature_base_block, self).__init__()

        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        residual = self.conv_skip(x)
        out = self.layers(x) + residual

        out = self.pooling(out)

        return out


class Pyramid_feature(nn.Module):
    def __init__(self):
        super(Pyramid_feature, self).__init__()

        self.layer0 = feature_base_block(in_channels=3, out_channels=8)
        self.layer1 = feature_base_block(in_channels=8, out_channels=16)
        self.layer2 = feature_base_block(in_channels=16, out_channels=32)
        self.layer3 = feature_base_block(in_channels=32, out_channels=64)

    def forward(self, x):

        pyramid = []

        x_down2 = self.layer0(x)
        pyramid.append(x_down2)
        x_down4 = self.layer1(x_down2)
        pyramid.append(x_down4)
        x_down8 = self.layer2(x_down4)
        pyramid.append(x_down8)
        x_down16 = self.layer3(x_down8)
        pyramid.append(x_down16)

        return pyramid


class MSHEModel(MultiScaleNet):
    def __init__(self, global_corr_type='global_corr', normalize='relu_l2norm',
                 local_corr_type='local_corr', md=4, upfeat_channels=2, batch_norm=True,
                 normalize_features=True, cyclic_consistency=True,
                 gocor_local_arguments=None, gocor_global_arguments=None):

        params = set_parameters(global_corr_type=global_corr_type, normalize=normalize,
                                md=md, local_corr_type=local_corr_type,
                                nbr_upfeat_channels=upfeat_channels, batch_norm=batch_norm,
                                normalize_features=normalize_features,
                                cyclic_consistency=cyclic_consistency,
                                gocor_local_arguments=gocor_local_arguments,
                                gocor_global_arguments=gocor_global_arguments)

        super().__init__(params)

        # level 4, 16x16
        decoder4 = FlowEstimatorResidualConnection(in_channels=256, batch_norm=True)

        self.decoder4 = decoder4
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # level 3, 32x32
        decoder3 = FlowEstimatorResidualConnection(in_channels=(2*self.params.md+1)**2 + 2,
                                                   batch_norm=True)
        self.decoder3 = decoder3

        self.adaptive_reso_refinement_module = adaptive_reso_refinement_module(34, self.params.batch_norm)

        # level 2, 1/8 of original resolution
        decoder2 = FlowEstimatorResidualConnection(in_channels=(2*self.params.md+1)**2 + 2,
                                                   batch_norm=True)
        self.decoder2 = decoder2
        self.upfeat2 = deconv(32, self.params.nbr_upfeat_channels, kernel_size=4, stride=2, padding=1)
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # level 1, 1/4 of original resolution
        decoder1 = FlowEstimatorResidualConnection(in_channels=(2*self.params.md+1)**2 + 4,
                                                   batch_norm=True)
        self.decoder1 = decoder1
        self.last_level_refinement_module = last_level_refinement_module(34, self.params.batch_norm)

        # initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

        self.initialize_global_corr()
        self.initialize_local_corr()

        self.share_features = Pyramid_feature()

    def forward(self, im_target, im_source, im_target_256, im_source_256, im_target_pyr=None, im_source_pyr=None,
                im_target_pyr_256=None, im_source_pyr_256=None):
        # im1 is target image, im2 is source image
        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()
        div = 1.0

        im_target_pyr = self.share_features(im_target)
        im_target_pyr_256 = self.share_features(im_target_256)
        im_source_pyr = self.share_features(im_source)
        im_source_pyr_256 = self.share_features(im_source_256)

        # 0:down2, 1:down4, 2:down8, 3:down16

        c11 = im_target_pyr[1]  # load_size original_res/4 x original_res/4
        c21 = im_source_pyr[1]
        c12 = im_target_pyr[2]  # load_size original_res/8 x original_res/8
        c22 = im_source_pyr[2]

        c13 = im_target_pyr_256[2]  # load_size 256/8
        c23 = im_source_pyr_256[2]
        c14 = im_target_pyr_256[3]  # load_size 256/16
        c24 = im_source_pyr_256[3]
        # RESOLUTION 256x256
        # level 4: 16x16
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)

        corr4 = self.get_global_correlation(c14, c24)

        x4, flow4 = self.decoder4(corr4)
        flow4 = unormalise_flow_or_mapping(flow4)

        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        up_flow4 = self.deconv4(flow4)

        # level 3: 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)

        corr3 = self.local_corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        corr3 = torch.cat((corr3, up_flow4), 1)

        x3, res_flow3 = self.decoder3(corr3)

        input_refinement = res_flow3 + up_flow4
        x3 = torch.cat((x3, input_refinement), 1)
        x_, res_flow3_ = self.adaptive_reso_refinement_module(x3)
        res_flow3 = res_flow3 + res_flow3_

        flow3 = res_flow3 + up_flow4

        up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
        up_flow3[:, 0, :, :] *= float(w_original) / float(256)
        up_flow3[:, 1, :, :] *= float(h_original) / float(256)

        # level 2 : 1/8 of original resolution
        ratio = 1.0 / 8.0
        warp2 = self.warp(c22, up_flow3*div*ratio)

        corr2 = self.local_corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        corr2 = torch.cat((corr2, up_flow3), 1)

        x2, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        up_flow2 = self.deconv2(flow2)

        up_feat2 = self.upfeat2(x2)

        # level 1: 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = self.warp(c21, up_flow2*div*ratio)

        corr1 = self.local_corr(c11, warp1)
        corr1 = self.leakyRELU(corr1)
        corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)

        x, res_flow1 = self.decoder1(corr1)

        input_refinement = res_flow1 + up_flow2
        x = torch.cat((x, input_refinement), 1)
        x_, res_flow1_ = self.last_level_refinement_module(x)
        res_flow1 = res_flow1 + res_flow1_

        flow1 = res_flow1 + up_flow2

        # prepare output dict
        output = {'flow_estimates': [flow2, flow1]}
        output_256 = {'flow_estimates': [flow4, flow3]}
        return output_256, output


@model_constructor
def model_construct(global_corr_type='global_corr', normalize='relu_l2norm',
                    local_corr_type='local_corr', md=4, nbr_upfeat_channels=2,
                    batch_norm=True, normalize_features=True,
                    cyclic_consistency=True, gocor_local_arguments=None,
                    gocor_global_arguments=None):

    net = MSHEModel(global_corr_type=global_corr_type,
                    normalize=normalize,
                    local_corr_type=local_corr_type, md=md,
                    upfeat_channels=nbr_upfeat_channels, batch_norm=batch_norm,
                    normalize_features=normalize_features, cyclic_consistency=cyclic_consistency,
                    gocor_local_arguments=gocor_local_arguments,
                    gocor_global_arguments=gocor_global_arguments)
    return net