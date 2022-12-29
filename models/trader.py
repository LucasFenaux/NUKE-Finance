import numpy as np
import torch.nn as nn
import torch
from models.transformer import StockTransformer
import torch.optim as optim
import torch.nn.functional as F
from configs.trading_config import gamma, device, week_config, month_config, year_config, sequence_length, alpha
from utils.normalization import parallelized_normalization_trader, undo_normalization_trader
from datetime import timedelta
import time


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int = 16):
        super(FeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.linear_layer = nn.Linear(in_features=420, out_features=20)  # parameters for the linear layer assuming input_dim=15
        self.extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim*2, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=input_dim*2), nn.ReLU(),
            nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*4, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=input_dim * 4), nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(in_channels=input_dim*4, out_channels=input_dim*8, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=input_dim * 8), nn.ReLU(),
            nn.Conv1d(in_channels=input_dim * 8, out_channels=input_dim * 16, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=input_dim * 16), nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(in_channels=input_dim * 16, out_channels=input_dim * 8, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=input_dim * 8), nn.ReLU(),
            nn.Conv1d(in_channels=input_dim * 8, out_channels=input_dim * 4, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=input_dim * 4), nn.ReLU(),
            nn.Conv1d(in_channels=input_dim * 4, out_channels=input_dim * 4, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=input_dim * 4), nn.ReLU(),
        )

    def forward(self, x):
        # assumes 4D input, we go one batch at a time since the feature extractor only extracts one stock at a time
        features = []
        for i in range(x.size()[0]):
            out = torch.flatten(self.extractor(x[i]), 1)
            out = self.linear_layer(out)
            features.append(out)
        return torch.stack(features, dim=0)


class ActionMaker(nn.Module):
    def __init__(self, num_stocks: int, num_features: int, num_actions: int):
        super(ActionMaker, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.num_actions = num_actions
        self.layer1 = nn.Linear((num_stocks * num_features) + 1, num_stocks * int(num_features/2))
        self.layer2 = nn.Linear(num_stocks * int(num_features/2), num_stocks*num_actions)
        self.relu = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # assumes 4D input, we go one batch at a time since the feature extractor only extracts one stock at a time
        out = self.relu(self.layer1(x))
        out = self.layer2(out)
        return out


class Trader(nn.Module):
    def __init__(self, device: torch.device, num_actions: int,
                 num_inputs: int = 1000, input_dim: int = 5, sequence_length: int = 100, pre_computed: bool = True):
        super(Trader, self).__init__()
        self.device = device
        self.num_inputs = num_inputs
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_actions = num_actions
        self.pre_computed = pre_computed  # whether the predictions are already computed
        if not pre_computed:
            self.init_predictors()
        self.feature_extractor = FeatureExtractor(input_dim*3).to(device)
        self.action_maker = ActionMaker(num_stocks=num_inputs, num_features=21, num_actions=num_actions).to(device)

    def init_predictors(self):
        # should be inorder week month year
        self.predictors = [StockTransformer(input_dim=self.input_dim, d_model=64, nhead=8, batch_first=True).to(device),
                           StockTransformer(input_dim=self.input_dim, d_model=64, nhead=8, batch_first=True).to(device),
                           StockTransformer(input_dim=self.input_dim, d_model=64, nhead=8, batch_first=True).to(device)]
        self.predictors[0].load_state_dict(
            torch.load(f"./saved_models/{self.predictors[0].get_name()}_{week_config[0]}_{week_config[1]}_week_"
                       f"{week_config[2]}_{week_config[3]}_{week_config[4]}.pth"))
        self.predictors[1].load_state_dict(
            torch.load(f"./saved_models/{self.predictors[1].get_name()}_{month_config[0]}_{month_config[1]}_month_"
                       f"{month_config[2]}_{month_config[3]}_{month_config[4]}.pth"))
        self.predictors[2].load_state_dict(
            torch.load(f"./saved_models/{self.predictors[2].get_name()}_{year_config[0]}_{year_config[1]}_year_"
                       f"{year_config[2]}_{year_config[3]}_{year_config[4]}.pth"))

    def pre_compute_predictions(self, week_data, month_data, year_data, start_device, end_device):
        # data is in the format entire horizon x num stocks x features
        start_time = time.monotonic()

        self.init_predictors()

        predicted_week_prices = []
        predicted_month_prices = []
        predicted_year_prices = []
        self.predictors[0] = self.predictors[0].to(end_device)
        self.predictors[1] = self.predictors[1].to(end_device)
        self.predictors[2] = self.predictors[2].to(end_device)

        self.predictors[0].eval()
        self.predictors[1].eval()
        self.predictors[2].eval()

        week_data = torch.Tensor(week_data).to(end_device)
        month_data = torch.Tensor(month_data).to(end_device)
        year_data = torch.Tensor(year_data).to(end_device)

        if isinstance(week_data, torch.Tensor):
            week_size = week_data.size()
            month_size = month_data.size()
            year_size = year_data.size()
        elif isinstance(week_data, np.ndarray):
            week_size = week_data.shape
            month_size = month_data.shape
            year_size = year_data.shape
        else:
            print(f"Week data type: {type(week_data)} not supported ")
            raise NotImplementedError

        with torch.no_grad():
            for j in range(3):
                self.predictors[j] = self.predictors[j].to(start_device)
                if j == 0:
                    print(f"pre computing week predictions")
                    size = week_size
                elif j == 1:
                    print(f"pre computing month predictions")
                    size = month_size
                elif j == 2:
                    print(f"pre computing year predictions")
                    size = year_size
                for i in range(sequence_length, size[0]):
                    if j == 0:
                        batched_week_data = week_data[i-sequence_length:i].to(start_device)
                        normalized_batched_week_data, min_vals, max_vals = parallelized_normalization_trader(batched_week_data)
                        normalized_predicted_week_data = self.predictors[0](normalized_batched_week_data.transpose(0, 1)
                                                                            ).transpose(0, 1)
                        predicted_week_data = undo_normalization_trader(normalized_predicted_week_data, min_vals[:, 0, :].squeeze(),
                                                                        max_vals[:, 0, :].squeeze()).to(end_device)
                        predicted_week_prices.append(predicted_week_data[-1])

                    elif j == 1:
                        batched_month_data = month_data[i-sequence_length:i].to(start_device)
                        normalized_batched_month_data, min_vals, max_vals = parallelized_normalization_trader(batched_month_data)
                        normalized_predicted_month_data = self.predictors[1](normalized_batched_month_data.transpose(0, 1)
                                                                            ).transpose(0, 1)
                        predicted_month_data = undo_normalization_trader(normalized_predicted_month_data, min_vals[:, 0, :].squeeze(),
                                                                        max_vals[:, 0, :].squeeze()).to(end_device)
                        predicted_month_prices.append(predicted_month_data[-1])

                    elif j == 2:
                        batched_year_data = year_data[i-sequence_length:i].to(start_device)
                        normalized_batched_year_data, min_vals, max_vals = parallelized_normalization_trader(batched_year_data)
                        normalized_predicted_year_data = self.predictors[2](normalized_batched_year_data.transpose(0, 1)
                                                                            ).transpose(0, 1)
                        predicted_year_data = undo_normalization_trader(normalized_predicted_year_data, min_vals[:, 0, :].squeeze(),
                                                                        max_vals[:, 0, :].squeeze()).to(end_device)
                        predicted_year_prices.append(predicted_year_data[-1])

                if j == 0:
                    predicted_week_prices = torch.stack(predicted_week_prices, dim=0).to(end_device)
                elif j == 1:
                    predicted_month_prices = torch.stack(predicted_month_prices, dim=0).to(end_device)
                elif j == 2:
                    predicted_year_prices = torch.stack(predicted_year_prices, dim=0).to(end_device)

                # we then move the predictors back to cpu to save gpu memory
                self.predictors[j] = self.predictors[j].to(end_device)
                torch.cuda.empty_cache()

        end_time = time.monotonic()
        print(f"Precomputation took: {timedelta(seconds=end_time - start_time)}")

        return predicted_week_prices, predicted_month_prices, predicted_year_prices

    def compute_predictions(self, week_data, month_data, year_data, start_device, end_device):
        # cpu = torch.device("cpu")
        predicted_week_prices = []
        predicted_month_prices = []
        predicted_year_prices = []
        self.predictors[0] = self.predictors[0].to(end_device)
        self.predictors[1] = self.predictors[1].to(end_device)
        self.predictors[2] = self.predictors[2].to(end_device)

        with torch.no_grad():
                self.predictors[0] = self.predictors[0].to(start_device)
                batched_week_data = week_data.to(start_device)
                predicted_week_prices.append(
                    self.predictors[0](batched_week_data.transpose(2, 1)).transpose(2, 1).to(end_device))
                predicted_week_prices = torch.cat(predicted_week_prices, dim=0).to(end_device)

                self.predictors[0] = self.predictors[0].to(end_device)

                self.predictors[1] = self.predictors[1].to(start_device)

                batched_month_data = month_data.to(start_device)
                predicted_month_prices.append(
                    self.predictors[1](batched_month_data.transpose(2, 1)).transpose(2, 1).to(end_device))
                predicted_month_prices = torch.cat(predicted_month_prices, dim=0).to(end_device)

                self.predictors[1] = self.predictors[1].to(end_device)

                self.predictors[2] = self.predictors[2].to(start_device)

                batched_year_data = year_data.to(start_device)
                predicted_year_prices.append(
                    self.predictors[2](batched_year_data.transpose(2, 1)).transpose(2, 1).to(end_device))
                predicted_year_prices = torch.cat(predicted_year_prices, dim=0).to(end_device)

                torch.cuda.empty_cache()

        return predicted_week_prices, predicted_month_prices, predicted_year_prices

    def forward(self, state, liquidity, portfolio):
        # state has size batch_size x num_stocks x input_features (prices = 15) x sequence length
        price_state = state.to(self.device).transpose(3, 2)
        # portfolio = state[:, :, 0]
        # # we need to shift and pad the portfolio since we will predict one forward
        # portfolio = torch.cat([portfolio[:, :, 1:],
        #                        torch.ones((portfolio.size()[0], portfolio.size()[1], 1)).to(self.device)*-1.], dim=2)

        batched_week_prices = price_state[:, :, :5]
        batched_month_prices = price_state[:, :, 5:10]
        batched_year_prices = price_state[:, :, 10:]
        if not self.pre_computed:
            predicted_week_prices, predicted_month_prices, \
            predicted_year_prices = self.compute_predictions(batched_week_prices, batched_month_prices,
                                                                 batched_year_prices, start_device=self.device,
                                                                 end_device=self.device)
        else:
            predicted_week_prices = batched_week_prices
            predicted_month_prices = batched_month_prices
            predicted_year_prices = batched_year_prices

        price_state = torch.cat([predicted_week_prices, predicted_month_prices, predicted_year_prices], dim=2)
                                 # portfolio.unsqueeze(2)], dim=2)
        price_state.requires_grad = True
        features = self.feature_extractor(price_state)

        # we now need to pad up to num_inputs number of stocks and shuffle them
        to_pad = self.num_inputs - features.size()[1]
        assert to_pad >= 0
        features = torch.cat([features, portfolio.unsqueeze(2)], dim=2)
        padding = torch.ones(features.size()[0], to_pad, features.size()[2]).to(self.device) * -1.
        padding.requires_grad = False
        features = torch.cat([features, padding], dim=1)

        # we now need to shuffle
        idx = torch.randperm(features.size()[1])
        idx_inv = torch.sort(idx).indices
        features = features[:, idx]
        liquidity.requires_grad = True
        liquidity_tensor = torch.ones((features.size()[0])).to(self.device) * liquidity.to(self.device)
        features = torch.flatten(features, start_dim=1)
        features = torch.cat([liquidity_tensor.unsqueeze(1), features], dim=1)

        qvalues = self.action_maker(features)
        qvalues = qvalues.reshape(qvalues.size()[0], -1, self.num_actions)
        # we now unshuffle the actions
        # qvalues should be batch_size x num_stocks x num_actions
        qvalues = qvalues[:, idx_inv]

        # and we remove the padded actions
        qvalues = qvalues[:, :self.num_inputs - to_pad]

        return qvalues

    def get_optimizer(self, feature_extractor_lr: float, action_maker_lr: float):
        return optim.Adam([{'params': self.feature_extractor.parameters(), 'lr': feature_extractor_lr},
                           {'params': self.action_maker.parameters(), 'lr': action_maker_lr}])

    def get_action(self, state, liquidity, portfolio):
        state = state
        qvalue = self.forward(state, liquidity, portfolio)

        _, actions = torch.max(qvalue, 2)
        return actions

    def train(self, mode: bool = True):
        self.feature_extractor.train(mode)
        self.action_maker.train(mode)
        if not self.pre_computed:
            for model in self.predictors:
                model.eval()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.stack(batch.action).contiguous().to(device)
        rewards = torch.Tensor(batch.reward).to(device)
        masks = torch.Tensor(batch.mask).to(device)
        liquidity = torch.Tensor(batch.liquidity).to(device)
        new_liquidity = torch.Tensor(batch.new_liquidity).to(device)
        portfolio = torch.stack(batch.portfolio).to(device)
        new_portfolio = torch.stack(batch.new_portfolio).to(device)

        pred = online_net(states, liquidity, portfolio)
        next_pred = target_net(next_states, new_liquidity, new_portfolio)

        pred = torch.sum(pred.mul(actions), dim=2)
        next_pred = next_pred.max(2)[0]
        masks = masks.unsqueeze(1).expand(-1, next_pred.size()[1])
        rewards = rewards.unsqueeze(1).expand(-1, next_pred.size()[1])
        target = masks * gamma * next_pred
        target += rewards

        loss = F.mse_loss(pred, target.detach()) * alpha
        optimizer.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        return loss.item()


if __name__ == '__main__':
    x = torch.rand((1, 500, 15, 100)).to(torch.device("cuda"))
    f = Trader(device=torch.device("cuda"), num_actions=5).to(torch.device("cuda"))
    y = f(x, torch.Tensor([1.]).to(torch.device("cuda")), torch.rand((1, 500)).to(torch.device("cuda")))
    print(y.size())
    y = f.get_action(x, torch.Tensor([1.]).to(torch.device("cuda")), torch.rand((1, 500)).to(torch.device("cuda")))
    print(y.size())
