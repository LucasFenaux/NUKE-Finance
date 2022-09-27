from trainer.vanilla_trainer import VanillaTrainer
from data.pre_process_data import get_week_data
from models.lstm import LSTM
from torch.nn import Transformer
import torch
from utils.losses import LSTMLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_weeks = 50

    train_loader, test_loader = get_week_data(num_weeks, batch_size=4096, keep_masked=True)

    # model = LSTM(num_inputs=6, num_outputs=6).to(device)
    model = Transformer(d_model=4, nhead=1, batch_first=True).to(device)

    model.device = device
    model.loss = LSTMLoss()

    trainer = VanillaTrainer(model)

    trainer.train(epochs=5, data_loader=train_loader)

    trainer.test(data_loader=test_loader)


if __name__ == '__main__':
    main()
