"""
PPO with Random-Network-Distillation
– random initialisiertes Net, that creates Feature-Vecs φ(s)

Run:
    python ppo_rnd.py --total_timesteps 1_000_000 --wandb_tracking True --video True
"""

import os
import random
import time
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

gym.register_envs(ale_py)



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    env_id: str = "ALE/SpaceInvaders-v5"
    eval_episodes: int = 100
    wandb_tracking: bool = True
    wandb_proj: str = "rle-rnd"
    wandb_entity: str = None
    video: bool = True
    torch_deterministic: bool = True
    cuda: bool = True
    save_model: bool = True
    eval_checkpoint: str = None
    total_timesteps: int = 1_000_000
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
    target_kl: float = None
    skip_frames: int = 4

    # --- RND specific ---
    # weighting of intrinsic reward in return
    int_coef: float = 1.0
    # coefficient for predictor MSE los
    predictor_coef: float = 10.0


    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


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


class RunningMeanStd:
    """Tracks mean and variance of a data stream."""
    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RNDNetwork(nn.Module):
    """Convolutional feature extractor used for both target and predictor."""

    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, output_dim), std=1.0),
        )

    def forward(self, x: torch.Tensor):
        # expects x in [0,1] range
        return self.features(x)


class RNDAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # shared feature extractor for policy/value
        self.backbone = nn.Sequential(
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
        n_actions = envs.single_action_space.n
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        # separate critics for extrinsic and intrinsic values
        self.ext_critic = layer_init(nn.Linear(512, 1), std=1.0)
        self.int_critic = layer_init(nn.Linear(512, 1), std=1.0)

        # RND networks
        self.rnd_target = RNDNetwork().eval()  # fixed random network
        for p in self.rnd_target.parameters():
            p.requires_grad = False
        self.rnd_predictor = RNDNetwork()


    def get_action_and_value(self, x, action=None):
        h = self.backbone(x / 255.0)
        logits = self.actor(h)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        ext_value = self.ext_critic(h)
        int_value = self.int_critic(h)
        return action, probs.log_prob(action), probs.entropy(), ext_value.squeeze(-1), int_value.squeeze(-1)

    def get_features(self, x):
        # for RND networks
        return self.rnd_target(x / 255.0), self.rnd_predictor(x / 255.0)


if __name__ == "__main__":
    args = tyro.cli(Args)
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
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = RNDAgent(envs).to(device)
    optimizer = optim.Adam(
        list(agent.backbone.parameters())
        + list(agent.actor.parameters())
        + list(agent.ext_critic.parameters())
        + list(agent.int_critic.parameters())
        + list(agent.rnd_predictor.parameters()),
        lr=args.lr,
        eps=1e-5,
    )

    # storage
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.uint8).to(device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_rew_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_rew_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_val_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_val_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # running stats for intrinsic reward normalisation
    int_rms = RunningMeanStd()

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.uint8).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for g in optimizer.param_groups:
                g["lr"] = frac * args.lr

        # rollout
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs

            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, entropy, ext_val, int_val = agent.get_action_and_value(next_obs.float(), None)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            ext_val_buf[step] = ext_val
            int_val_buf[step] = int_val

            # step envs
            next_obs_cpu, ext_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.logical_or(torch.from_numpy(terminations),
                                         torch.from_numpy(truncations)).to(device, dtype=torch.float32)
            next_obs = torch.tensor(next_obs_cpu, dtype=torch.uint8).to(device)

            # intrinsic reward
            with torch.no_grad():
                tgt_f, pred_f = agent.get_features(next_obs.float())
                int_reward = torch.square(pred_f - tgt_f).mean(dim=1)
            int_rms.update(int_reward.cpu())
            int_reward /= torch.sqrt(int_rms.var.to(device) + 1e-8)

            # store rewards
            ext_rew_buf[step] = torch.tensor(ext_reward).to(device).view(-1)
            int_rew_buf[step] = int_reward

            if (terminations.any() or truncations.any()) and "episode" in infos:
                for info_idx in np.argwhere(infos["_episode"]):
                    writer.add_scalar("train/episodic_return", infos["episode"]["r"][info_idx], global_step)
                    writer.add_scalar("train/episodic_length", infos["episode"]["l"][info_idx], global_step)
                    writer.add_scalar("train/episodic_time", infos["episode"]["t"][info_idx], global_step)

        # compute returns and advantages separately
        with torch.no_grad():
            next_ext_val, next_int_val = agent.get_action_and_value(next_obs.float(), None)[3:5]

            ext_adv_buf = torch.zeros_like(ext_rew_buf).to(device)
            int_adv_buf = torch.zeros_like(int_rew_buf).to(device)
            lastgaelam_ext = lastgaelam_int = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues_ext = next_ext_val
                    nextvalues_int = next_int_val
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues_ext = ext_val_buf[t + 1]
                    nextvalues_int = int_val_buf[t + 1]

                delta_ext = ext_rew_buf[t] + args.gamma * nextvalues_ext * nextnonterminal - ext_val_buf[t]
                delta_int = int_rew_buf[t] + args.gamma * nextvalues_int * nextnonterminal - int_val_buf[t]
                ext_adv_buf[t] = lastgaelam_ext = (
                    delta_ext + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_ext
                )
                int_adv_buf[t] = lastgaelam_int = (
                    delta_int + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_int
                )
            ext_returns = ext_adv_buf + ext_val_buf
            int_returns = int_adv_buf + int_val_buf
            total_adv = ext_adv_buf + args.int_coef * int_adv_buf

        # flatten
        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape).float()
        b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_total_adv = total_adv.reshape(-1)
        b_ext_values = ext_val_buf.reshape(-1)
        b_int_values = int_val_buf.reshape(-1)

        # optimization
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = b_obs[mb_inds].to(device)

                _, newlogprob, entropy, new_ext_val, new_int_val = agent.get_action_and_value(
                    mb_obs, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_total_adv[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value losses
                ext_v_loss = 0.5 * ((new_ext_val - b_ext_returns[mb_inds]) ** 2).mean()
                int_v_loss = 0.5 * ((new_int_val - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss

                entropy_loss = entropy.mean()

                # predictor loss (update rnd_predictor only)
                tgt_f, pred_f = agent.get_features(mb_obs)
                predictor_loss = torch.square(pred_f - tgt_f).mean()

                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + args.predictor_coef * predictor_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred_ext, y_true_ext = b_ext_values.cpu().numpy(), b_ext_returns.cpu().numpy()
        var_y = np.var(y_true_ext)
        explained_var_ext = np.nan if var_y == 0 else 1 - np.var(y_true_ext - y_pred_ext) / var_y

        # logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/ext_value_loss", ext_v_loss.item(), global_step)
        writer.add_scalar("losses/int_value_loss", int_v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/predictor_loss", predictor_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance_ext", explained_var_ext, global_step)
        writer.add_scalar("charts/StepPerSecond", int(global_step / (time.time() - start_time)), global_step)

        print(
            f"Iteration {iteration}/{args.num_iterations} | SPS: {int(global_step / (time.time() - start_time))} | "
            f"ExtVLoss: {ext_v_loss.item():.3f} | Predictor: {predictor_loss.item():.3f}"
        )

    # save model
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    else:
        model_path = args.eval_checkpoint

    # evaluation (extrinsic reward only)
    episodic_events = evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=f"{run_name}-eval",
        Model=RNDAgent,
        device=device,
    )

    for idx, event in enumerate(episodic_events):
        writer.add_scalar("eval/episodic_return", event["return"], idx)
        writer.add_scalar("eval/episodic_length", event["length"], idx)
        writer.add_scalar("eval/episodic_time", event["time"], idx)

    try:
        writer.close()
        envs.close()
    except Exception:
        pass
