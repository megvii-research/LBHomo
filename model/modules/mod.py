import torch
import torch.nn as nn
import numpy as np


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, batch_norm=False, relu=True):
    if batch_norm:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.LeakyReLU(0.1, inplace=True))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes))
    else:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias),
                                nn.LeakyReLU(0.1))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias))


def predict_flow(in_planes, nbr_out_channels=2):
    return nn.Conv2d(in_planes, nbr_out_channels, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
    mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow


def subspace_project(input, vectors):

    b_, c_, h_, w_ = input.shape
    basis_vector_num = vectors.shape[1]
    V_t = vectors.view(b_, basis_vector_num, h_ * w_)
    V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
    V = V_t.permute(0, 2, 1)
    mat = torch.bmm(V_t, V)
    mat_inv = torch.inverse(mat)
    project_mat = torch.bmm(mat_inv, V_t)
    input_ = input.view(b_, c_, h_ * w_)
    project_feature = torch.bmm(project_mat, input_.permute(0, 2, 1))
    output = torch.bmm(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)

    return output


class Subspace(nn.Module):

    def __init__(self, ch_in, k=16, use_SVD=True, use_PCA=False):

        super(Subspace, self).__init__()
        self.k = k
        self.Block = SubspaceBlock(ch_in, self.k)
        self.use_SVD = use_SVD
        self.use_PCA = use_PCA

    def forward(self, x):

        sub = self.Block(x)
        x = subspace_project(x, sub)

        return x


class SubspaceBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(SubspaceBlock, self).__init__()

        self.conv0 = conv(inplanes, planes, kernel_size=1, stride=1, dilation=1, padding=0, batch_norm=True)
        self.conv1 = conv(planes, planes, kernel_size=1, stride=1, dilation=1, padding=0, batch_norm=True)
        self.conv2 = conv(planes, planes, kernel_size=1, stride=1, dilation=1, padding=0, batch_norm=True)

    def forward(self, x):

        residual = self.conv0(x)

        out = self.conv1(residual)

        out = self.conv2(out)

        out = out + residual

        return out


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super(Res_block, self).__init__()

        self.conv0 = conv(in_channels, out_channels, kernel_size=3, stride=1, batch_norm=bn, relu=False)
        self.conv1 = conv(out_channels, out_channels, kernel_size=3, stride=1, batch_norm=bn, relu=False)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=1, batch_norm=bn, relu=False)

        self.skip = conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, batch_norm=bn,
                         relu=False, bias=False)
        self.act_layer = nn.LeakyReLU(0.1)

    def forward(self, x):
        x0 = self.conv0(x)
        x_skip = self.skip(x)

        x1 = self.conv1(self.act_layer(x0))

        x2 = self.conv2(self.act_layer(x1)) + x_skip

        x_out = self.act_layer(x2)

        return x_out


class FlowEstimatorResidualConnection(nn.Module):

    def __init__(self, in_channels, batch_norm):
        super(FlowEstimatorResidualConnection, self).__init__()

        self.block0 = Res_block(in_channels, 128, batch_norm)  # 32 * 32
        self.block1 = Res_block(128, 96, batch_norm)  # 32 * 32

        self.subspace = Subspace(96)

        self.block2 = Res_block(96, 64, batch_norm)  # 16 * 16
        self.block3 = Res_block(64, 32, batch_norm)  # 16 * 16

        self.predict_flow = predict_flow(32)

    def forward(self, x):
        b, _, h, w = x.shape

        x_0 = self.block0(x)  # 32 * 32
        x_1 = self.block1(x_0)  # 32 * 32

        x_1 = self.subspace(x_1)

        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)

        flow = self.predict_flow(x_3)

        return x_3, flow


class last_level_refinement_module(nn.Module):
    def __init__(self, input_channels, batch_norm=True):
        super(last_level_refinement_module, self).__init__()

        self.conv1 = conv(input_channels, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                          batch_norm=batch_norm)
        self.conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.subspace = Subspace(96)
        self.conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.conv7 = predict_flow(32)

    def forward(self, x):
        x = self.subspace(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        x = self.conv6(self.conv5(x))
        res = self.conv7(x)

        return x, res


class adaptive_reso_refinement_module(nn.Module):
    def __init__(self, input_channels, batch_norm=True):
        super(adaptive_reso_refinement_module, self).__init__()

        self.conv1 = conv(input_channels, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                             batch_norm=batch_norm)
        self.conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.subspace = Subspace(96)
        self.conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.conv7 = predict_flow(32)

    def forward(self, x):
        x = self.subspace(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        x = self.conv6(self.conv5(x))
        res = self.conv7(x)

        return x, res


