"""Evaluates the model"""

import argparse
import logging
import os
import cv2, imageio

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn
import torch.nn.functional as F
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from utils_operations.pixel_wise_mapping import warp
from loss.loss import test_model_on_image_pair
from loss.error_compute import compute_error, identity_error
from common.manager import Manager

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model0', help="Directory containing params.json")
parser.add_argument('--restore_file', default='experiments/base_model0/model_ep9.pth', help="name of the file in --model_dir containing weights to load")


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    print("eval begin!")

    # loss status and eval status initial
    manager.reset_loss_status()
    manager.reset_metric_status(manager.params.eval_type)
    torch.cuda.empty_cache()
    model.eval()

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107']
    LT = ['0000038', '0000044', '00000238', '00000177', '00000188', '00000181', '00000239']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    with torch.no_grad():

        for data_batch in manager.dataloaders[manager.params.eval_type]:
            video_name = data_batch["video_name"][-1]
            npy_path = data_batch["points_path"][-1]
            npy_name = data_batch["npy_name"][-1]
            input_images = data_batch["ori_images"].cuda()

            point_dic = np.load(npy_path, allow_pickle=True)

            p = []
            pt_pairs = point_dic[0]
            dist_pairs = point_dic[1]
            for j in range(len(point_dic[0])):
                p.append([(point_dic[0][j][0], point_dic[0][j][1]),
                          (point_dic[1][j][0], point_dic[1][j][1])])

            source = input_images[:, :3, :, :]
            target = input_images[:, 3:, :, :]

            flow_b = test_model_on_image_pair(source, target, model)
            flow_f = test_model_on_image_pair(target, source, model)

            error = compute_error(flow_f, flow_b, p)
            error_identity = identity_error(pt_pairs, dist_pairs)
            if error > error_identity:
                error = error_identity
            print('{}:{}'.format(npy_name, error))
            if video_name in RE:
                MSE_RE.append(error)
            elif video_name in LT:
                MSE_LT.append(error)
            elif video_name in LL:
                MSE_LL.append(error)
            elif video_name in SF:
                MSE_SF.append(error)
            elif video_name in LF:
                MSE_LF.append(error)

        MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
        MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
        MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
        MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
        MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
        MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_LF_avg + MSE_SF_avg) / 5

        Metric = {"MSE_avg": MSE_avg, "MSE_RE_avg": MSE_RE_avg, "MSE_LT_avg": MSE_LT_avg, "MSE_LL_avg": MSE_LL_avg,
                  "MSE_SF_avg": MSE_SF_avg, "MSE_LF_avg": MSE_LF_avg}
        manager.update_metric_status(metrics=Metric, split=manager.params.eval_type, batch_size=1)

        # update data to logger
        manager.logger.info("Loss/valid epoch_{} {}: AVG:{:.2f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} "
                            .format(manager.params.eval_type, manager.epoch_val, MSE_avg,
                                    MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

        # For each epoch, print the metric
        manager.print_metrics(manager.params.eval_type, title=manager.params.eval_type, color="green")

        manager.epoch_val += 1
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        model.train()
        val_metrics = {'MSE_avg': MSE_avg}

        return val_metrics


def eval_save_result(save_file, save_name, manager):

    # save dir: model_dir
    save_dir_gif = os.path.join(manager.params.model_dir, 'gif')
    if not os.path.exists(save_dir_gif):
        os.makedirs(save_dir_gif)

    save_dir_gif_epoch = os.path.join(save_dir_gif, str(manager.epoch_val))
    if not os.path.exists(save_dir_gif_epoch):
        os.makedirs(save_dir_gif_epoch)

    if type(save_file)==list: # save gif
        utils.create_gif(save_file, os.path.join(save_dir_gif_epoch, save_name))
    elif type(save_file)==str: # save string information
        f = open(os.path.join(save_dir_gif_epoch, save_name), 'w')
        f.write(save_file)
        f.close()
    elif manager.val_img_save: # save single image
        cv2.imwrite(os.path.join(save_dir_gif_epoch, save_name), save_file)


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(model=model, optimizer=None, scheduler=None, params=params, dataloaders=dataloaders, writer=None, logger=logger)

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    evaluate(model, manager)
