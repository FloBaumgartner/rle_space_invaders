"""
PPO with IMPALA‑CNN‑Backbone
– bigger, residual‑based CNN architecture

Run:
    python ppo_impala.py --total_timesteps 1_000_000 --wandb_tracking True --video True
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
    wandb_proj: str = "rle-impala"
    wandb_entity: str = None
    #capture videos of the agents performances
    video: bool = True
    # if true, torch.backends.cudnn.deterministic=False
    torch_deterministic: bool = True
    # cuda support
    cuda: bool = True
    # save model to `runs/{run_name}`
    save_model: bool = True
    # evaluation of the checkpoint
    eval_checkpoint: str = None
    # timesteps of the experiments
    total_timesteps: int = 5_000_000
    # learning rate
    lr: float = 2.5e-4
    # the number of parallel game environments
    num_envs: int = 16
    # number of steps to run in each env per policy rollout
    num_steps: int = 128
    # learning rate annealing for policy and value networks
    anneal_lr: bool = True
    # discount factor
    gamma: float = 0.99
    # lambda for the general advantage estimation
    gae_lambda: float = 0.95
    # number of mini-batches
    num_minibatches: int = 4
    # epochs to update policy
    update_epochs: int = 4
    # advantages normalization
    norm_adv: bool = True
    # surrogate clipping coefficient
    clip_coef: float = 0.1
    # use a clipped loss for the value function
    clip_vloss: bool = True
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


    #batch size (computed in runtime)
    batch_size: int = 0
    #mini batch size (computed in runtime)
    minibatch_size: int = 0
    # number of iterations (computed in runtime)
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
        #env.reset(seed=seed)
        return env

    return thunk


class ImpalaCNNAgent(nn.Module):
    """3‑Block IMPALA CNN Head."""

    def __init__(self, envs, in_channels=4):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

        self.trunk = nn.Sequential(block(in_channels, 16), block(16, 32), block(32, 32))
        self.flat_dim = 32 * 11 * 11
        self.fc = nn.Linear(self.flat_dim, 256)
        self.actor = nn.Linear(256, envs.single_action_space.n)
        self.critic = nn.Linear(256, 1)

    def features(self, x):
        x = x / 255.0  # scale
        x = self.trunk(x)
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))

    def get_value(self, x):
        return self.critic(self.features(x))

    def get_action_and_value(self, x, action=None):
        h = self.features(x)
        logits = self.actor(h)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(h)


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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = ImpalaCNNAgent(envs).to(device)
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
    episodic_events = evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=f"{run_name}-eval",
        Model=ImpalaCNNAgent,
        device=device,
    )

    for idx, event in enumerate(episodic_events):
        writer.add_scalar("eval/episodic_return", event['return'], idx)
        writer.add_scalar("eval/episodic_length", event['length'], idx)
        writer.add_scalar("eval/episodic_time", event['time'], idx)

    # try/Except because of AttributeError (RecordVideo object has no attribute 'enabled') at evaluation
    try:
        writer.close()
        envs.close()
    except:
        pass




