import os
import random
import time
import copy
from typing import Dict, Any
from dataclasses import dataclass

import gymnasium as gym

import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from ppo_eval import evaluate

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Optuna imports (Bayesian / TPE sampler)
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler



gym.register_envs(ale_py)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    env_id: str = "ALE/SpaceInvaders-v5"
    eval_episodes: int = 8
    wandb_tracking: bool = False  # off by default for HPO -> turned on in the final run
    wandb_proj: str = "rle-hpo"
    wandb_entity: str | None = None
    video: bool = False
    torch_deterministic: bool = True
    cuda: bool = True
    save_model: bool = True
    eval_checkpoint: str | None = None
    total_timesteps: int = 400_000

    # these hyper‑parameters will be overwritten by Optuna
    lr: float = 2.5e-4
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    skip_frames: int = 4

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # ---------------- Optuna flags ------------------
    tune: bool = True  # if True: run Bayesian HPO, else vanilla training
    n_trials: int = 64  # number of HPO trials
    study_name: str = "ppo_spaceinvaders_tpe"
    storage: str | None = None

def make_env(env_id, idx, video, run_name):
    def thunk():
        if video and idx == 0:
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


# Single‑run training (factorised function)
def train_once(args: Args) -> Dict[str, Any]:
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y%m%d_%H%M%S')}"

    if args.wandb_tracking:
        import wandb

        wandb.init(
            project=args.wandb_proj,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    if not args.eval_checkpoint:
        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.lr
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)


                if (terminations.any() or truncations.any()) and "episode" in infos:
                    for info in np.argwhere(infos["_episode"]):
                        print(f"global_step={global_step}, episodic_return={infos['episode']['r'][info]}")
                        writer.add_scalar("train/episodic_time", infos["episode"]["t"][info], global_step)
                        writer.add_scalar("train/episodic_return", infos["episode"]["r"][info], global_step)
                        writer.add_scalar("train/episodic_length", infos["episode"]["l"][info], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                                  b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)


            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/StepPerSecond", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)

            print(f"model saved to {model_path}")

    model_path = args.eval_checkpoint if args.eval_checkpoint else model_path
    eval_events = evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=f"{run_name}-eval",
        Model=Agent,
        device=device,
        video=False,
    )
    returns = [e["return"] for e in eval_events]
    mean_return = float(np.mean(returns))

    writer.add_hparams(
        {
            "lr": args.lr,
            "num_steps": args.num_steps,
            "clip_coef": args.clip_coef,
            "gae_lambda": args.gae_lambda,
        },
        {"mean_return": mean_return},
    )

    writer.close()
    envs.close()

    return {
        "mean_return": mean_return,
        "model_path": model_path,
        "run_name": run_name,
    }


# Optuna objective & optimisation driver
def suggest_and_train(trial: optuna.Trial, base_args: Args) -> float:
    """Optuna objective: sample hyper‑params ➜ call `train_once` ➜ return score."""

    args = copy.deepcopy(base_args)
    # sample H‑params
    args.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    args.num_steps = trial.suggest_categorical("num_steps", [64, 128, 256])
    args.clip_coef = trial.suggest_float("clip_coef", 0.05, 0.3)
    args.gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)

    # Entropy & value‑loss coefficients
    args.ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
    args.vf_coef = trial.suggest_float("vf_coef", 0.2, 1.0)

    # -------------------------------------
    metrics = train_once(args)

    # Report the metric back to Optuna & allow pruning
    trial.report(metrics["mean_return"], step=0)
    return metrics["mean_return"]

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.tune:
        # "direction='maximize'" because we want high episodic return.
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            sampler=TPESampler(seed=args.seed),
            pruner=MedianPruner(),
        )
        study.optimize(lambda t: suggest_and_train(t, args), n_trials=args.n_trials, show_progress_bar=True)

        print("\n===== Hyper‑parameter optimisation finished =====")
        print("Best mean return:", study.best_value)
        print("Best hyper‑parameters:\n", study.best_params)

        #retrain with best params for longer total_timesteps
        best_args = copy.deepcopy(args)
        best_args.tune = False  # regular training now
        best_args.lr = study.best_params["lr"]
        best_args.num_steps = study.best_params["num_steps"]
        best_args.clip_coef = study.best_params["clip_coef"]
        best_args.gae_lambda = study.best_params["gae_lambda"]
        best_args.ent_coef = study.best_params["ent_coef"]
        best_args.vf_coef = study.best_params["vf_coef"]
        best_args.wandb_tracking = True
        best_args.total_timesteps = 5_000_000  # retrain for longer
        best_args.eval_episodes = 100
        best_args.video = True  # record video for final agent
        train_once(best_args)
    else:
        # Plain vanilla training identical to ppo_initial
        train_once(args)