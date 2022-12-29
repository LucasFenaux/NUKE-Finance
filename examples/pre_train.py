import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from trainer.vanilla_trainer import VanillaTrainer
from data.pre_process_data import get_un_indexed_data
from models.lstm import LSTM
from torch.nn import Transformer
from models.transformer import StockTransformer
import torch
from utils.losses import LSTMLoss
from torch.optim import Adam, SGD
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="train a model on a dataset.")
    parser.add_argument("-d", "--data_type", type=str, default="week", choices=["week", "month", "year"])
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-i", "--num_increments", type=int, default=50)
    parser.add_argument("-s", "--sequence_length", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("-b", "--batch_size", type=int, default=2048)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--input_dim", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true", dest="use_cache")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_increments = args.num_increments
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    sequence_length = args.sequence_length
    data_type = args.data_type
    input_dim = args.input_dim
    use_cache = args.use_cache

    train_loader, test_loader = get_un_indexed_data(num_increments, sequence_length=sequence_length,
                                                    data_type=data_type, batch_size=batch_size, overwrite=False,
                                                    input_dim=input_dim, use_cache=use_cache)

    # model = LSTM(num_inputs=6, num_outputs=6).to(device)
    # model = Transformer(d_model=5, nhead=1, batch_first=True).to(device)
    model = StockTransformer(input_dim=input_dim, d_model=64, nhead=8, batch_first=True).to(device)
    model.device = device
    model.loss = LSTMLoss(alpha=0.1, beta=0.1)

    trainer = VanillaTrainer(model)

    # optimizer = Adam(lr=0.01, params=model.parameters())
    optimizer = SGD(lr=lr, params=model.parameters(), momentum=0.9, weight_decay=1e-4, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer.train(epochs=epochs, data_loader=train_loader, optimizer=optimizer, scheduler=scheduler)

    trainer.test(data_loader=test_loader)

    if args.save:
        torch.save(model.state_dict(), f"./saved_models/{model.get_name()}_{epochs}_{lr}_{data_type}_{sequence_length}_"
                                       f"{num_increments}_{input_dim}.pth")


if __name__ == '__main__':
    main()
