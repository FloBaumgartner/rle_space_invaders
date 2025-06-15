import os
import time
import random
from dataclasses import dataclass

import gymnasium as gym
import ale_py
import numpy as np
import torch
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable, Dict, List, Tuple, Type
from torch import nn

gym.register_envs(ale_py)


@dataclass
class Args:
    # name and configuration
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # seed
    seed: int = 1
    # environment configuration
    env_id: str = "ALE/SpaceInvaders-v5"
    # number of evaluation episodes
    eval_episodes: int = 100
    # track with WandB
    wandb_tracking: bool = True
    # WandB configurations
    wandb_proj: str = "rle-reccurrent"
    wandb_entity: str = None
    # capture videos of the agents performances
    video: bool = False
    # cuda support
    #cuda: bool = True
    # save model to `runs/{run_name}`
    save_model: bool = True
    # evaluation of the checkpoint
    eval_checkpoint: str = None
    # timesteps of the experiments
    total_timesteps: int = 5_000_000
    # learning rate
    lr: float = 2.5e-4
    # the number of parallel game environments
    num_envs: int = 4  # fewer envs because we store LSTM states
    # number of steps to run in each env per policy rollout
    num_steps: int = 128 #rollout length (BPTT truncation length)
    #minibatch bsize for PPO update
    batch_size: int = num_envs * num_steps
    # learning rate annealing for policy and value networks
    # anneal_lr: bool = True
    # discount factor
    gamma: float = 0.99
    # lambda for the general advantage estimation
    gae_lambda: float = 0.95
    # number of mini-batches
    # num_minibatches: int = 4
    # epochs to update policy
    # update_epochs: int = 4
    # advantages normalization
    # norm_adv: bool = True
    # surrogate clipping coefficient
    # clip_coef: float = 0.1
    # use a clipped loss for the value function
    # clip_vloss: bool = True
    # coeff of the entropy
    ent_coef: float = 0.01
    # coeff of the value function
    vf_coef: float = 0.5
    # max norm for gradient clipping
    max_grad_norm: float = 0.5
    # target KL divergence threshold
    target_kl: float = None
    # frames to skip in MaxAndSkipEnv wrapper
    skip_frames: int = 4

    # new
    n_epochs: int = 2
    clip_range: float = 0.1


def make_env(env_id, idx, video, run_name):
    def thunk():
        if video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=args.skip_frames)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk

class ImpalaBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.pool(x)
        return x


class ImpalaCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=512)
        c = observation_space.shape[0]
        self.net = nn.Sequential(
            ImpalaBlock(c, 32),
            ImpalaBlock(32, 64),
            ImpalaBlock(64, 64),
            nn.AdaptiveAvgPool2d(1),  # 64 × 1 × 1
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs / 255.0)





def evaluate(
    model_path: str,
    make_env_fn: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    device: torch.device = torch.device("cpu"),
    video: bool = True,
):
    env = gym.vector.SyncVectorEnv([make_env_fn(env_id, 0, video, run_name)])
    model = RecurrentPPO.load(model_path, device=device)
    obs, _ = env.reset()
    lstm_state = None  # type: ignore
    done = np.zeros((env.num_envs,), dtype=bool)
    returns: List[float] = []
    lengths: List[int] = []
    times: List[float] = []

    print("Starting evaluation…")
    while len(returns) < eval_episodes:
        action, lstm_state = model.predict(obs, state=lstm_state, episode_start=done, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)
        if done[0] and "episode" in info:
            ep_info = info["episode"]
            returns.append(ep_info["r"][0])
            lengths.append(ep_info["l"][0])
            times.append(ep_info["t"][0])
            print(f"eval_episode={len(returns)}, return={ep_info['r'][0]}")
    print("…finished evaluation")
    env.close()
    return [{"return": r, "length": l, "time": t} for r, l, t in zip(returns, lengths, times)]


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y%m%d_%H%M%S')}"
    if args.wandb_tracking:
        import wandb

        wandb.init(
            project=args.wandb_proj,
            entity=args.wandb_entity,
            name=run_name,
            monitor_gym=True,
            config=vars(args),
            sync_tensorboard=True,
            save_code=True,
        )


    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # vectorised envs
    env_fns = [make_env(args.env_id, i, args.video, run_name) for i in range(args.num_envs)]

    envs = SubprocVecEnv(env_fns, start_method="spawn")
    envs = VecFrameStack(envs, n_stack=1)  # already stacked 4 frames via wrapper → keep 1 here

    # policy & model
    policy_kwargs: Dict = dict(
        features_extractor_class=ImpalaCNN,
        lstm_hidden_size=64,
        share_features_extractor=True,
    )

    model = RecurrentPPO(
        policy="CnnLstmPolicy",
        env=envs,
        learning_rate=args.lr,
        n_steps=args.num_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs",
        seed=args.seed,
        device=device,
    )


    # evaluation callback (no video during training)
    eval_env = make_env(args.env_id, 999, False, run_name)() # () for type gym.Env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"runs/{run_name}",
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.total_timesteps // 10,
        deterministic=True,
        render=False,
    )

    # training
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=eval_callback)

    # save final model
    model_path = f"runs/{run_name}/recurrent_ppo"
    if args.save_model:
        model.save(model_path)
        print(f"Model saved to {model_path}.zip")

    # final evaluation with video
    evaluate(
        model_path=model_path,
        env_id=args.env_id,
        make_env_fn=make_env,
        eval_episodes=args.eval_episodes,
        run_name=f"{run_name}-eval",
        device=device
    )

    envs.close()


