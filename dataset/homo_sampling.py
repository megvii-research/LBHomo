import torch
import numpy as np
from utils_operations.flow_and_mapping_operations import unormalise_and_convert_mapping_to_flow
from utils_operations.homography_parameters_sampling import RandomHomography
from torch.nn.modules.module import Module


def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.size())
    sz[dim] = desired_dim_len
    return tensor.expand(tuple(sz))


def homography_mat_from_4_pts(theta):
    b = theta.size(0)
    if not theta.size() == (b, 8):
        theta = theta.view(b, 8)
        theta = theta.contiguous()

    xp = theta[:, :4].unsqueeze(2)
    yp = theta[:, 4:].unsqueeze(2)

    x = torch.FloatTensor([-1, -1, 1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    y = torch.FloatTensor([-1, 1, -1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    z = torch.zeros(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    o = torch.ones(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    single_o = torch.ones(1).unsqueeze(1).unsqueeze(0).expand(b, 1, 1)

    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()

    A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2),
                   torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h = torch.bmm(torch.inverse(A[:, :, :8]), -A[:, :, 8].unsqueeze(2))
    # add h33
    h = torch.cat([h, single_o], 1)

    H = h.squeeze(2)

    return H


class HomographyGridGen(Module):
    """Dense correspondence map generator, corresponding to a homography transform."""

    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(HomographyGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: load_size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X.requires_grad = False
        self.grid_Y.requires_grad = False
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def forward(self, theta):
        b = theta.size(0)
        if theta.size(1) == 9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        h0 = H[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1 = H[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2 = H[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3 = H[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4 = H[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5 = H[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6 = H[:, 6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7 = H[:, 7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8 = H[:, 8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)

        grid_Xp = grid_X * h0 + grid_Y * h1 + h2
        grid_Yp = grid_X * h3 + grid_Y * h4 + h5
        k = grid_X * h6 + grid_Y * h7 + h8

        grid_Xp /= k
        grid_Yp /= k

        return torch.cat((grid_Xp, grid_Yp), 3)


class Synthetic_Homo:
    def __init__(self, img_size=(360, 640), random_hom=0.4, use_cuda=False):
        if not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        self.out_h, self.out_w = img_size

        self.random_t_hom = random_hom
        # for tps
        self.use_cuda = use_cuda
        self.homo_grid_sample = HomographyGridGen(out_h=self.out_h, out_w=self.out_w, use_cuda=use_cuda)

    def __call__(self, *args, **kwargs):
        theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
        theta_hom = theta_hom + (np.random.rand(8) - 0.5) * 2 * self.random_t_hom
        theta_hom = torch.Tensor(theta_hom.astype(np.float32)).unsqueeze(0)
        theta_hom = theta_hom.cuda() if self.use_cuda else theta_hom

        mapping = self.homo_grid_sample.forward(theta_hom)
        flow_gt = unormalise_and_convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 2xhw

        return flow_gt.requires_grad_(False)


class Synthetic_HomoFlow:
    def __init__(self, img_size, sampling_module=None):
        if not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        self.img_size = img_size

        if sampling_module is None:
            sampling_module = Synthetic_Homo(img_size=self.img_size,
                                             random_hom=0.35)
            # this is quite strong transformations
        self.sample = sampling_module

    def __call__(self, training=True, *args, **kwargs):

        flow_gt = self.sample()

        return flow_gt.requires_grad_(False)


class GetRandomSyntheticHomographyFlow:
    def __init__(self, size_output_flow, homo_sampling_module=None):
        """
        Args:
            settings:
            size_output_flow:
            homo_sampling_module: module to sample the homogaphy transform parameters.
                                  If None, we use the default module.
        """

        if homo_sampling_module is None:
            homo_sampling_module = RandomHomography(p_flip=0.0, max_rotation=1.0, max_shear=0.1,
                                                    max_scale=0.1, max_ar_factor=0.1,
                                                    min_perspective=-0.0005, max_perspective=0.0005,
                                                    max_translation=2, pad_amount=0)
        self.homography_transform = homo_sampling_module

        if not isinstance(size_output_flow, tuple):
            size_output_flow = (size_output_flow, size_output_flow)
        self.size_output_flow = size_output_flow

    def __call__(self, training=True, *args, **kwargs):

        with torch.no_grad():
            do_flip, rot, shear_values, scale_factors, perpective_factor, tx, ty = self.homography_transform.roll()
            H = self.homography_transform._construct_t_mat(self.size_output_flow, do_flip, rot,
                                                           shear_values, scale_factors, tx=tx, ty=ty,
                                                           perspective_factor=perpective_factor)

        return H / H[2, 2]


def convert_mapping_to_flow(mapping, output_channel_first=True):
    if not isinstance(mapping, np.ndarray):
        # torch tensor
        if len(mapping.shape) == 4:
            if mapping.shape[1] != 2:
                # load_size is BxHxWx2
                mapping = mapping.permute(0, 3, 1, 2)

            B, C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(0, 2, 3, 1)
        else:
            if mapping.shape[0] != 2:
                # load_size is HxWx2
                mapping = mapping.permute(2, 0, 1)

            C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()

            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(1, 2, 0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(mapping.shape) == 4:
            if mapping.shape[3] != 2:
                # load_size is Bx2xHxW
                mapping = mapping.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = mapping.shape[:3]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0, 3, 1, 2)
        else:
            if mapping.shape[0] == 2:
                # load_size is 2xHxW
                mapping = mapping.transpose(1, 2, 0)
            # HxWx2
            h_scale, w_scale = mapping.shape[:2]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:, :, 0] = mapping[:, :, 0] - X
            flow[:, :, 1] = mapping[:, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)


def from_homography_to_pixel_wise_mapping(shape, H):
    """
    From a homography relating image I to image I', computes pixel wise mapping and pixel wise displacement
    between pixels of image I to image I'
    Args:
        shape: shape of image
        H: homography

    Returns:
        map_x mapping of each pixel of image I in the horizontal direction (given index of its future position)
        map_y mapping of each pixel of image I in the vertical direction (given index of its future position)
    """
    h_scale, w_scale = shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    X, Y = X.flatten(), Y.flatten()
    # X is same shape as shape, with each time the horizontal index of the pixel

    # create matrix representation --> each contain horizontal coordinate, vertical and 1
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

    # multiply Hinv to XYhom to find the warped grid
    XYwarpHom = np.dot(H, XYhom)
    Xwarp = XYwarpHom[0, :] / (XYwarpHom[2, :] + 1e-8)
    Ywarp = XYwarpHom[1, :] / (XYwarpHom[2, :] + 1e-8)

    # reshape to obtain the ground truth mapping
    map_x = Xwarp.reshape((h_scale, w_scale))
    map_y = Ywarp.reshape((h_scale, w_scale))
    return map_x.astype(np.float32), map_y.astype(np.float32)

