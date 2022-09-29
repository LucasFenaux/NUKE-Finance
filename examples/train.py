from trainer.vanilla_trainer import VanillaTrainer
from data.pre_process_data import get_unindexed_week_data
from models.lstm import LSTM
from torch.nn import Transformer
from models.transformer import StockTransformer
import torch
from utils.losses import LSTMLoss

# TODO: fix normalization to not take into account last value
# TODO: fix guiding loss term to only care about the difference w.r.t. the last value
# TODO: implement the other time horizons


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_weeks = 50

    train_loader, test_loader = get_unindexed_week_data(num_weeks, batch_size=8192, overwrite=False)

    # model = LSTM(num_inputs=6, num_outputs=6).to(device)
    # model = Transformer(d_model=5, nhead=1, batch_first=True).to(device)
    model = StockTransformer(input_dim=5, d_model=64, nhead=8, batch_first=True).to(device)
    model.device = device
    model.loss = LSTMLoss(alpha=0.001, beta=0.001)

    trainer = VanillaTrainer(model)

    trainer.train(epochs=5, data_loader=train_loader)

    trainer.test(data_loader=test_loader)


if __name__ == '__main__':
    main()
