import torch.nn.functional as F
import torch


class MultiScaleFlow:

    def __init__(self, level_weights, loss_function, downsample_gt_flow):

        self.level_weights = level_weights
        self.loss_function = loss_function
        self.downsample_gt_flow = downsample_gt_flow

    def one_scale(self, est_flow, gt_flow, mask=None):
        if self.downsample_gt_flow:
            b, _, h, w = est_flow.size()
            gt_flow = F.interpolate(gt_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
                if mask.shape[2] != h or mask.shape[3] != w:
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).byte()
                    mask = mask.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask.byte()
        else:
            b, _, h, w = gt_flow.shape
            est_flow = F.interpolate(est_flow, (h, w), mode='bilinear', align_corners=False)
            if mask is not None:
                mask = mask.unsqueeze(1)
                if mask.shape[2] != h or mask.shape[3] != w:
                    mask = F.interpolate(mask.float(), (h, w), mode='bilinear', align_corners=False).byte()
                    mask = mask.bool() if float(torch.__version__.split('.')[1]) >= 1 else mask.byte()

        return self.loss_function(est_flow, gt_flow, mask=mask)

    def __call__(self, network_output, gt_flow, mask=None):

        if isinstance(network_output, dict):
            flow_output = network_output['flow_estimates']
        else:
            flow_output = network_output
        if type(flow_output) not in [tuple, list]:
            flow_output = [flow_output]
        assert(len(self.level_weights) == len(flow_output))

        loss = 0
        for i, (flow, weight) in enumerate(zip(flow_output, self.level_weights)):
            b, _, h, w = flow.shape
            if mask is not None and isinstance(mask, list):
                mask_used = mask[i]
            else:
                mask_used = mask
            level_loss = weight * self.one_scale(flow, gt_flow, mask=mask_used)
            loss += level_loss
        return loss
