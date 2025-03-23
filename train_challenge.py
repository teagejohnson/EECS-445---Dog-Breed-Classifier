"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
import torch
import numpy as np
import random
from dataset_challenge import get_train_val_test_loaders
from model.challenge import Challenge
from train_common_challenge import *
from utils import config
import utils

from train_target import freeze_layers
import copy

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train(tr_loader, va_loader, te_loader, model, model_name, num_layers=0):
    """Train transfer learning model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)

    print("Loading target model with", num_layers, "layers frozen")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = utils.make_training_plot("Target Training")

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][2]

    #TODO: patience for early stopping
    patience = 20
    curr_patience = 0
    #

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        epoch += 1

    print("Finished Training")

    # Keep plot open
    utils.save_tl_training_plot(num_layers)
    utils.hold_training_plot()


def main():
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"), augment = True
        )
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"),
        )
    # Model
    freeze_none = Challenge()

    # TODO: define loss function, and optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
    
    # # Attempts to restore the latest checkpoint if exists
    # print("Loading challenge...")
    # model, start_epoch, stats = restore_checkpoint(model, config("challenge.checkpoint"))

    # axes = utils.make_training_plot()

    # # Evaluate the randomly initialized model
    # evaluate_epoch(
    #     axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    # )

    # # initial val loss for early stopping
    # prev_val_loss = stats[0][1]

    # #TODO: define patience for early stopping
    # patience = 10
    # curr_patience = 0
    # #

    # # Loop over the entire dataset multiple times
    # epoch = start_epoch
    # while curr_patience < patience:
    #     # Train model
    #     train_epoch(tr_loader, model, criterion, optimizer)

    #     # Evaluate model
    #     evaluate_epoch(
    #         axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
    #     )

    #     # Save model parameters
    #     save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)

    #     # Updates early stopping parameters
    #     curr_patience, prev_val_loss = early_stopping(
    #         stats, curr_patience, prev_val_loss
    #     )
    #     #
    #     epoch += 1
    # print("Finished Training")
    # # Save figure and keep plot open
    # utils.save_challenge_training_plot()
    # utils.hold_training_plot()

    freeze_none, _, _ = restore_checkpoint(
        freeze_none, './checkpoints/source_challenge/', force=True, pretrain=True
    )

    freeze_one = copy.deepcopy(freeze_none)
    freeze_two = copy.deepcopy(freeze_none)
    freeze_three = copy.deepcopy(freeze_none)

    freeze_layers(freeze_one, 1)
    freeze_layers(freeze_two, 2)
    freeze_layers(freeze_three, 3)

    train(tr_loader, va_loader, te_loader, freeze_none, "./checkpoints/challenge0/", 0)
    train(tr_loader, va_loader, te_loader, freeze_one, "./checkpoints/challenge1/", 1)
    train(tr_loader, va_loader, te_loader, freeze_two, "./checkpoints/challenge2/", 2)
    train(tr_loader, va_loader, te_loader, freeze_three, "./checkpoints/challenge3/", 3)


if __name__ == "__main__":
    main()
