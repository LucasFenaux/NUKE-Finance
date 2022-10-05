import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from examples.train import parse_args
from models.transformer import StockTransformer
import torch
from trainer.vanilla_trainer import VanillaTrainer
from data.pre_process_data import get_un_indexed_data
from utils.losses import LSTMLoss


def load_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_increments = args.num_increments
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    sequence_length = args.sequence_length
    data_type = args.data_type
    input_dim = args.input_dim

    model = StockTransformer(input_dim=input_dim, d_model=64, nhead=8, batch_first=True).to(device)
    model.device = device
    model.loss = LSTMLoss(alpha=0.1, beta=0.1)

    _, test_loader = get_un_indexed_data(num_increments, sequence_length=sequence_length,
                                         data_type=data_type, batch_size=batch_size, overwrite=False,
                                         input_dim=input_dim)

    model.load_state_dict(torch.load(f"./saved_models/{model.get_name()}_{epochs}_{lr}_{data_type}_{sequence_length}_"
                                     f"{num_increments}_{input_dim}.pth"))

    trainer = VanillaTrainer(model)
    trainer.test(data_loader=test_loader)


def main():
    args = parse_args()
    load_and_eval(args)


if __name__ == '__main__':
    main()