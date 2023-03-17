"""Train the model"""

import argparse
import datetime
import os

import torch
import torch.optim as optim
from tqdm import tqdm
# from apex import amp

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.loss import compute_losses

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default='experiments/base_model/test_model_best_5.6313.pth',
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  #
parser.add_argument('--only_weights', default=True,
                    help='Only use weights to load or load all train status.')


def train(model, manager):

    # loss status initial
    manager.reset_loss_status()

    # set model to training mode
    torch.cuda.empty_cache()
    model.train()

    # Use tqdm for progress bar
    with tqdm(total=len(manager.dataloaders['train'])) as t:
        for i, data_batch in enumerate(manager.dataloaders['train']):

            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            # compute model output and loss
            loss = compute_losses(data_batch, model, manager.params)

            manager.update_loss_status(loss=loss, split="train")

            # clear previous gradients, compute gradients of all variables loss
            manager.optimizer.zero_grad()
            loss['total'].backward()

            # performs updates using calculated gradients
            manager.optimizer.step()

            manager.update_step()
            if i % manager.params.eval_freq == 0 and i != 0:
                # print('\n')
                val_metrics = evaluate(model, manager)
                avg = val_metrics['MSE_avg']
                manager.cur_val_score = avg
                manager.check_best_save_last_checkpoints(latest_freq=1)

            # infor print
            print_str = manager.print_train_info()

            t.set_description(desc=print_str)
            t.update()

    manager.scheduler.step()

    # update epoch: epoch += 1
    manager.update_epoch()


def train_and_evaluate(model, manager):

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()

    for epoch in range(manager.params.num_epochs):

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)
        evaluate(model, manager)
        manager.check_best_save_last_checkpoints(latest_freq=1)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Update args into params
    params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'train.log'))

    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the input data pipeline
    logger.info("Loading the train datasets from {}".format(params.data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

    # initial status for checkpoint manager
    manager = Manager(model=model, optimizer=optimizer, scheduler=scheduler, params=params, dataloaders=dataloaders, writer=None, logger=logger)

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, manager)
