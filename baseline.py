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
    # WandB configuration
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
    """
    Generische Evaluations-Funktion, die sowohl RandomAgent als auch HeuristicAgent
    unterstützt. Gibt pro Episode ein Dictionary mit folgenden Keys zurück:
      - 'return': kumulierter Reward (nach Clipping)
      - 'length': Anzahl der Schritte in der Episode
      - 'time': Dauer (in Sekunden) für die Episode
      - 'shots_fired': Anzahl der Schuss-Ereignisse (Action==FIRE oder FIRE-Kombination)
      - 'hits': Anzahl der Rewards > 0 (also tatsächlich erreichte Punkte)
      - 'avg_reward_per_hit': Mittelwert reward/hit (falls hits>0, sonst 0)
    """
    # Wir erstellen genau eine Vector-Env-Instanz (SyncVectorEnv), idx=0
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, video, run_name)])
    action_space = envs.single_action_space

    # Reset der Vector-Env
    obs, _ = envs.reset()
    episodic_events = []

    # tracking variables
    shots_fired = 0
    hits = 0
    total_reward = 0.0      # Summe of Rewards in the current episode
    ep_steps = 0            # count of steps in the current episode

    completed_episodes = 0
    ep_start_time = time.time()

    while completed_episodes < eval_episodes:
        # Agent chooses an action
        action = agent.get_action()
        if action in [1, 4, 5]:
            shots_fired += 1

        # Environment-Step
        next_obs, rewards, terminated, truncated, infos = envs.step([action])

        # update reward and step-count
        reward = float(rewards[0])
        total_reward += reward
        ep_steps += 1
        if reward > 0:
            hits += 1

        # check if Episode is done
        if terminated[0] or truncated[0]:
            ep_return = total_reward
            ep_length = ep_steps
            ep_time = time.time() - ep_start_time
            avg_rph = (ep_return / hits) if hits > 0 else 0.0

            # print to Terminal
            print(
                f"[{agent.__class__.__name__}] "
                f"|||Episode={completed_episodes}|| Return={ep_return:.0f}|| Length={ep_length}|||"
                f"|||Shots={shots_fired}|| Hits={hits}|| AvgR/Hit={avg_rph:.2f}|| Time={ep_time:.2f}s|||"
            )

            # svae
            episodic_events.append({
                'return': ep_return,
                'length': ep_length,
                'time': ep_time,
                'shots_fired': shots_fired,
                'hits': hits,
                'avg_reward_per_hit': avg_rph,
            })

            # End episode: reset environment
            completed_episodes += 1
            obs, _ = envs.reset()
            shots_fired = 0
            hits = 0
            total_reward = 0.0
            ep_steps = 0
            ep_start_time = time.time()

        # update observation
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

    # Baseline 1: RandomAgent
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
        writer.add_scalar("random/shots_fired", event['shots_fired'], idx)
        writer.add_scalar("random/hits", event['hits'], idx)
        writer.add_scalar("random/avg_reward_per_hit", event['avg_reward_per_hit'], idx)

    # 4.2 Baseline 2: HeuristicAgent
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
        writer.add_scalar("heuristic/shots_fired", event['shots_fired'], idx)
        writer.add_scalar("heuristic/hits", event['hits'], idx)
        writer.add_scalar("heuristic/avg_reward_per_hit", event['avg_reward_per_hit'], idx)

    writer.close()