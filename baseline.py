import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import ale_py
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter

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
    wandb_proj: str = "rle-baselines"
    wandb_entity: str = None
    #capture videos of the agents performances
    video: bool = True



def make_env(env_id, idx, video, run_name):
    def thunk():
        if video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self):
        return self.action_space.sample()


class HeuristicAgent:
    """
    Ein einfacher regelbasierter Agent für SpaceInvaders:
    - Feuert in jeder Zeiteinheit (FIRE).
    - Bewegt sich in einer Richtung (links oder rechts), wechselt jedoch
      nach einer festen Anzahl von Schritten (bounce_steps) die Richtung.
    - Nutzt direkt die Aktionscodes von ALE/SpaceInvaders (NOOP=0, FIRE=1, RIGHT=2, LEFT=3, RIGHTFIRE=4, LEFTFIRE=5).
    """
    def __init__(self, action_space: gym.Space, bounce_steps: int = 20):
        assert isinstance(action_space, gym.spaces.Discrete), "Erwartet diskretes Aktions-Set."
        # ALE/SpaceInvaders has 6 actions: [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
        self.bounce_steps = bounce_steps
        self.current_step = 0
        # Start-direction: 1 = to the right, -1 = to the left
        self.direction = 1
        # Mapping: direction= 1: 4 = RIGHTFIRE, direction= -1: 5 LEFTFIRE
        self._action_map = {1: 4, -1: 5}

    def get_action(self):
        # change direction after bounce_steps
        if self.current_step >= self.bounce_steps:
            self.current_step = 0
            self.direction *= -1
        action = self._action_map[self.direction]
        self.current_step += 1
        return action

def evaluate_agent(
    env_id: str,
    eval_episodes: int,
    run_name: str,
    agent,
    video: bool = True,
):

    # Wir erstellen genau eine Vector-Env-Instanz (SyncVectorEnv), idx=0
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, video, run_name)])
    action_space = envs.single_action_space

    # Reset der Vector-Env
    obs, _ = envs.reset()
    episodic_events = []


    completed_episodes = 0
    ep_start_time = time.time()

    while len(episodic_events) < eval_episodes:
        action = agent.get_action()
        next_obs, rewards, terminated, truncated, infos = envs.step([action])

        if (terminated or truncated) and "episode" in infos:
            print(
                f"eval_episode={len(episodic_events)}, episodic_return={infos['episode']['r']}, episodic_length={infos['episode']['l']}, episodic_time={infos['episode']['t']}")
            episodic_events += [
                {'return': infos['episode']['r'], 'length': infos['episode']['l'], 'time': infos['episode']['t']}]
        obs = next_obs

    return episodic_events



if __name__ == "__main__":
    args = tyro.cli(Args)

    # set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_name = f"{args.env_id}-{args.exp_name}-{args.seed}-{time.strftime('%Y%m%d_%H%M%S')}"
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

    # RandomAgent baseline
    print("Start Eval RandomAgent: -------------------------------------------")
    random_agent = RandomAgent(gym.make(args.env_id).action_space)
    random_events = evaluate_agent(
        args.env_id,
        args.eval_episodes,
        run_name + "_Random",
        agent=random_agent,
        video=args.video
    )

    # Log results RandomAgent in TensorBoard
    for idx, event in enumerate(random_events):
        writer.add_scalar("random/episodic_return", event['return'], idx)
        writer.add_scalar("random/episodic_length", event['length'], idx)
        writer.add_scalar("random/episodic_time", event['time'], idx)

    # HeuristicAgent baseline
    print("Start Eval HeuristicAgent: -------------------------------------------")
    heuristic_agent = HeuristicAgent(gym.make(args.env_id).action_space, bounce_steps=50)
    heuristic_events = evaluate_agent(
        args.env_id,
        args.eval_episodes,
        run_name + "_Heuristic",
        agent=heuristic_agent,
        video=args.video
    )

    # Log results HeuristicAgent in TensorBoard
    for idx, event in enumerate(heuristic_events):
        writer.add_scalar("heuristic/episodic_return", event['return'], idx)
        writer.add_scalar("heuristic/episodic_length", event['length'], idx)
        writer.add_scalar("heuristic/episodic_time", event['time'], idx)

    writer.close()