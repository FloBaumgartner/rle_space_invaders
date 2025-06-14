from typing import Callable

import gymnasium as gym
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0,video, run_name)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    ep_events = []

    print('Starting evaluation...')
    while len(ep_events) < eval_episodes:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        outputs = agent.get_action_and_value(obs_tensor)
        actions = outputs[0]
        next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        if (terminated or truncated) and "episode" in infos:
            print(
                f"eval_episode={len(ep_events)}, episodic_return={infos['episode']['r']}, episodic_length={infos['episode']['l']}, episodic_time={infos['episode']['t']}")
            ep_events += [
                {'return': infos['episode']['r'], 'length': infos['episode']['l'], 'time': infos['episode']['t']}]
        obs = next_obs

    print("...finished evaluation")

    try:
        envs.close()
    except:
        pass

    return ep_events

