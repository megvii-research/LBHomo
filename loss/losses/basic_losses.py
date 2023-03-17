import torch
import torch.nn.functional as F


class L1:
    """ Computes L1 loss. """
    def __init__(self, sum_normalized=True, ratio=1.0):
        """
        Args:
            sum_normalized: bool, compute the sum over tensor and divide by number of image pairs per batch?
            ratio:
        """
        super().__init__()
        self.sum_normalized = sum_normalized
        self.ratio = ratio
        self.valid_transformation_bool = None

    def __call__(self, gt_flow, est_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        L1 = torch.sum(torch.abs(est_flow-gt_flow), 1, keepdim=True)

        if mask is not None:
            mask = ~torch.isnan(L1.detach()) & ~torch.isinf(L1.detach()) & mask
        else:
            mask = ~torch.isnan(L1.detach()) & ~torch.isinf(L1.detach())
    
        if mask is not None:
            L1 = L1 * mask.float()
            L = 0
            for bb in range(0, b):
                L1[bb, ...][mask[bb, ...] == 0] = L1[bb, ...][mask[bb, ...] == 0].detach()
                norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                L = L + L1[bb][mask[bb, ...] != 0].sum() * norm_const
            if self.sum_normalized:
                return L / b
            else:
                return L
    
        if self.valid_transformation_bool is not None:
            L1 = L1 * self.valid_transformation_bool.float().unsqueeze(1).unsqueeze(1)
            # puts it to 0 in the case where there are no valid transformation
            return L1.sum()/self.valid_transformation_bool.sum()
    
        if self.sum_normalized:
            return L1.sum()/b
        else:
            return L1


class L1Charbonnier:
    """Computes L1 Charbonnier loss. """
    def __init__(self, sum_normalized=True, ratio=1.0):
        """
        Args:
            sum_normalized: bool, compute the sum over tensor and divide by number of image pairs per batch?
            ratio:
        """
        super().__init__()
        self.sum_normalized = sum_normalized
        self.ratio = ratio
        self.valid_transformation_bool = None

    def __call__(self, gt_flow, est_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        epsilon = 0.01
        alpha = 0.4
        L1 = torch.sum(torch.abs(est_flow - gt_flow), 1, keepdim=True)
        norm = torch.pow(L1 + epsilon, alpha)

        if mask is not None:
            mask = ~torch.isnan(norm.detach()) & ~torch.isinf(norm.detach()) & mask
        else:
            mask = ~torch.isnan(norm.detach()) & ~torch.isinf(norm.detach())
    
        if mask is not None:
            norm = norm * mask.float()
            L = 0
            for bb in range(0, b):
                norm[bb, ...][mask[bb, ...] == 0] = norm[bb, ...][mask[bb, ...] == 0].detach()
                norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                L = L + norm[bb][mask[bb, ...] != 0].sum() * norm_const
            if self.sum_normalized:
                return L / b
            else:
                return L
    
        if self.valid_transformation_bool is not None:
            norm = norm * self.valid_transformation_bool.float().unsqueeze(1).unsqueeze(1)
            # puts it to 0 in the case where there are no valid transformation
            return norm.sum()/self.valid_transformation_bool.sum()
    
        if self.sum_normalized:
            return norm.sum() / b
        else:
            return norm