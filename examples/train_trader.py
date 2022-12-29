import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from models.trader import Trader
from data.trading_data import get_training_trade_data, ExperienceReplay, TradingEnv
import numpy as np
import matplotlib.pyplot as plt
from configs.trading_config import *
from configs import trading_config
import pickle
import time
from datetime import timedelta
from utils import utils


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

    return parser


def train_one_epoch(online_trading_model, target_trading_model, env, steps, replay, epsilon, optimizer, epoch):
    online_trading_model.train()
    target_trading_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("score", utils.SmoothedValue(window_size=1, fmt="{value:.3e}"))
    metric_logger.add_meter("epsilon", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    score = 0
    count = 0
    model_portfolio, liquidity, state, episode_length = env.reset()
    header = f"Epoch: [{epoch}] |"
    action_count = None

    for _, _ in enumerate(metric_logger.log_every(range(episode_length), log_frequency, header)):
        steps += 1

        state = state.to(device).unsqueeze(0)
        liquidity = torch.Tensor([liquidity]).to(device)
        model_portfolio = model_portfolio.to(device).unsqueeze(0)

        action = get_action(state, liquidity, model_portfolio, online_trading_model, epsilon, env).long().squeeze()
        action_one_hot = torch.zeros(action.size()[0], num_actions).to(device).long()
        action_one_hot[torch.arange(0, action.size()[0]), action] = 1
        if action_count is None:
            action_count = action_one_hot
        else:
            action_count += action_one_hot
        new_model_portfolio, new_liquidity, next_state, reward, done = env.step(action)
        next_state = next_state.to(device).squeeze()

        mask = 0 if done else 1
        if not np.isnan(reward):
            replay.push(state.squeeze(), next_state, action_one_hot, reward, mask, liquidity,
                        new_liquidity, model_portfolio.squeeze(), new_model_portfolio.squeeze())

            score += reward
        state = next_state.clone().detach()
        model_portfolio = new_model_portfolio.clone().detach()
        liquidity = new_liquidity
        metric_logger.update(score=score, epsilon=epsilon)
        if steps > initial_exploration and len(replay) > batch_size:
            if steps % train_frequency == 0:
                epsilon -= 0.00002
                epsilon = max(epsilon, max_epsilon)

                batch = replay.sample(batch_size)
                loss = Trader.train_model(online_trading_model, target_trading_model, optimizer, batch)
                metric_logger.update(loss=loss)
                del batch
                count += 1

            if steps % update_target == 0:
                update_target_model(online_trading_model, target_trading_model)
        if done:
            break

    print(f"Cumulative Score: {score}")
    print(f"Mean Loss: {metric_logger.meters['loss'].global_avg:.3e}")

    return score, metric_logger.meters["loss"].global_avg, epsilon, action_count


def get_action(state, liquidity, portfolio, online_net, epsilon, env):

    if np.random.rand() <= epsilon:
        action = env.get_random_action()
    else:
        action = online_net.get_action(state, liquidity, portfolio)
    return action


def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    online_trading_model = Trader(device=device, num_actions=5, pre_computed=use_cached_pre_computing).to(device)
    target_trading_model = Trader(device=device, num_actions=5, pre_computed=use_cached_pre_computing).to(device)

    optimizer = online_trading_model.get_optimizer(feature_extractor_lr=fe_lr, action_maker_lr=am_lr)

    start_epoch = 0
    epsilon = 0.9
    steps = 0
    score_list = []
    action_count = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        config = checkpoint["config"]
        for key, val in config.items():
            exec(key + '=val')
        online_trading_model.load_state_dict(checkpoint["online_trading_model"])
        target_trading_model.load_state_dict(checkpoint["target_trading_model"])
        online_trading_model = online_trading_model.to(device)
        target_trading_model = target_trading_model.to(device)
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        replay = checkpoint["replay"]
        replay.move_to_device()
        env = checkpoint["env"]
        args = checkpoint["args"]
        steps = checkpoint["steps"]
        epsilon = checkpoint["epsilon"]
        # epsilon = 0.115
        score_list = checkpoint["score_list"]
        action_count = checkpoint["action_count"]
    else:
        ((weeks, week_index), (months, month_index), (years, year_index)), tickers = \
            get_training_trade_data(use_cache=use_cached_pre_processing)

        # we pre-compute the predictions on all the data to save time at live training
        if use_cached_pre_computing:
            if os.path.exists(os.path.join(save_dir, f"pre_computing_cache.pkl")):
                with open(save_dir + f"pre_computing_cache.pkl", "rb") as f:
                    pred_weeks, pred_months, pred_years = pickle.load(f)
            else:
                # we want to use cache but it doesn't exist
                print("Pre-computing wasn't cached")
                pred_weeks, pred_months, pred_years = online_trading_model.pre_compute_predictions(
                    week_data=weeks, month_data=months, year_data=years, start_device=device,
                    end_device=torch.device("cpu"))
                # we save to the cache
                with open(save_dir + f"pre_computing_cache.pkl", "wb") as f:
                    pickle.dump((pred_weeks, pred_months, pred_years), f)
        else:
            pred_weeks, pred_months, pred_years = online_trading_model.pre_compute_predictions(
                week_data=weeks, month_data=months, year_data=years, start_device=device,
                end_device=torch.device("cpu"))

        replay = ExperienceReplay(replay_capacity, device=device)
        env = TradingEnv(num_actions=num_actions, starting_liquidity=starting_liquidity, data=(weeks, months, years),
                         pred_data=(pred_weeks, pred_months, pred_years), tickers=tickers,
                         indices=(week_index, month_index, year_index), device=device)

    for e in range(start_epoch, epochs):
        score, loss, epsilon, actions = train_one_epoch(online_trading_model, target_trading_model, env, steps, replay,
                                                        epsilon, optimizer, e)
        actions = actions.detach().cpu().to(torch.float)
        actions_sum = torch.sum(actions, dim=1)
        scaled_actions = torch.zeros_like(actions)
        for i in range(actions.size()[0]):
            scaled_actions[i] = actions[i] / actions_sum[i]
        print(torch.mean(scaled_actions, dim=0))

        action_count.append(actions)
        score_list.append(score)

        if args.output_dir:
            config = vars(trading_config)

            excluded = set(['torch'])
            bad = lambda k: k in excluded or k.startswith('__')
            config = {k: v for k, v in config.items() if not bad(k)}
            checkpoint = {
                "online_trading_model": online_trading_model.state_dict(),
                "target_trading_model": target_trading_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
                "args": args,
                "env": env,
                "replay": replay,
                "config": config,
                "steps": steps,
                "score_list": score_list,
                "action_count": action_count,
                "epsilon": epsilon
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"checkpoint.pth"))

    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    moving_averages = []
    i = 0
    while i < len(score_list) - 100 + 1:
        this_window = score_list[i: i + 100]
        window_average = sum(this_window) / 100
        moving_averages.append(window_average)
        i += 1
    Ep_arr = np.array(moving_averages)
    plt.plot(Ep_arr)
    if args.output_dir:
        plt.savefig(os.path.join(args.output_dir, 'trader.png'))
        torch.save(online_trading_model.state_dict(), (os.path.join(args.output_dir,
                                                                    f"final_model.pth")))
    else:
        plt.savefig('trader.png')

    action_count = torch.sum(torch.stack(action_count, dim=0), dim=0)
    action_count_sum = torch.sum(action_count, dim=1)
    for i in range(action_count.size()[0]):
        action_count[i] /= action_count_sum[i]
    print(torch.mean(action_count, dim=0))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
