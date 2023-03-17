import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.consensus_network_modules import MutualMatching
from ..modules.feature_correlation_layer import FeatureL2Norm, GlobalFeatureCorrelationLayer
from third_party.GOCor.GOCor.global_gocor_modules import GlobalGOCorWithFlexibleContextAwareInitializer
from third_party.GOCor.GOCor import local_gocor
from third_party.GOCor.GOCor.optimizer_selection_functions import define_optimizer_local_corr


class MatchingNetParams:

    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):

        if len(default) > 1:
            raise ValueError('Can only give one default value.')

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        return hasattr(self, name)


class BaseMultiScaleNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = None

    @staticmethod
    def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        if float(torch.__version__.split('.')[1]) >= 3:
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)

        return output

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, *input):
        raise NotImplementedError


class MultiScaleNet(BaseMultiScaleNet):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = params
        self.visdom = None
        self.l2norm = FeatureL2Norm()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU(inplace=False)

    def initialize_global_corr(self):
        if self.params.global_corr_type == 'GlobalGOCor':
            # Global GOCor with FlexibleContextAware Initializer module
            self.corr = GlobalGOCorWithFlexibleContextAwareInitializer(
                global_gocor_arguments=self.params.GOCor_global_arguments)
        else:
            # Feature correlation layer
            self.corr = GlobalFeatureCorrelationLayer(shape='3D', normalization=False,
                                                      put_W_first_in_channel_dimension=False)

    def initialize_local_corr(self):
        initializer = local_gocor.LocalCorrSimpleInitializer()
        optimizer = define_optimizer_local_corr(self.params.GOCor_local_arguments)
        self.local_corr = local_gocor.LocalGOCor(filter_initializer=initializer, filter_optimizer=optimizer)

    def get_global_correlation(self, c14, c24):
        b = c14.shape[0]
        if 'GOCor' in self.params.global_corr_type:
            if self.params.normalize_features:
                corr4, losses4 = self.corr(self.l2norm(c14), self.l2norm(c24))
            else:
                corr4, losses4 = self.corr(c14, c24)
        else:
            # directly obtain the 3D correlation
            if self.params.normalize_features:
                corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            else:
                corr4 = self.corr(c24, c14)  # first source, then target

        if self.params.cyclic_consistency:
            # to add on top of the correlation ! (already included in NC-Net)
            corr4d = MutualMatching(corr4.view(b, c24.shape[2], c24.shape[3], c14.shape[2], c14.shape[3]).unsqueeze(1))
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])

        if self.params.normalize == 'l2norm':
            corr4 = self.l2norm(corr4)
        elif self.params.normalize == 'relu_l2norm':
            corr4 = self.l2norm(F.relu(corr4))
        elif self.params.normalize == 'leakyrelu':
            corr4 = self.leakyRELU(corr4)
        return corr4


def set_parameters(global_corr_type='global_corr', normalize='relu_l2norm',
                   local_corr_type='local_corr', md=4, nbr_upfeat_channels=2,
                   batch_norm=True, normalize_features=True, cyclic_consistency=True,
                   gocor_local_arguments=None, gocor_global_arguments=None):
    params = MatchingNetParams()
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.global_corr_type = global_corr_type
    params.normalize = normalize
    params.local_corr_type = local_corr_type
    params.batch_norm = batch_norm
    params.normalize_features = normalize_features
    params.cyclic_consistency = cyclic_consistency
    params.md = md
    params.nbr_upfeat_channels = nbr_upfeat_channels
    params.GOCor_local_arguments = gocor_local_arguments
    params.GOCor_global_arguments = gocor_global_arguments


    return params
