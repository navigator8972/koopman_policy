# Load the policy and the env in which it was trained
import time
import torch
import numpy as np

#from garage import rollout
from garage.experiment import Snapshotter
from garage.np import discount_cumsum, stack_tensor_dict_list

import argparse

def save_numpy_to_video_matplotlib(array, filename, interval=50):
    from matplotlib import animation
    from matplotlib import pyplot as plt

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    img = ax.imshow(array[0])

    def img_show(i):
        img.set_array(array[i])
        # print("updating image {}".format(i))
        return [img]

    ani = animation.FuncAnimation(fig, img_show, len(array), interval=interval)

    #ani.save('{}.gif'.format(filename), writer='imagemagick', fps=1000/interval)
    ani.save('{}.mp4'.format(filename))
    return

def rollout(env,
            agent,
            *,
            max_episode_length=np.inf,
            animated=False,
            video_name=None,
            pause_per_frame=None,
            deterministic=False,
            is_softgym=False):
    """Sample a single episode of the agent in the environment.
    Args:
        agent (Policy): Policy used to select actions.
        env (Environment): Environment to perform actions in.
        max_episode_length (int): If the episode reaches this many timesteps,
            it is truncated.
        animated (bool): If true, render the environment after each step.
        pause_per_frame (float): Time to sleep between steps. Only relevant if
            animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
    Returns:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape
                :math:`(T + 1, S^*)`, i.e. the unflattened observation space of
                    the current environment.
            * actions(np.array): Non-flattened array of actions. Should have
                shape :math:`(T, S^*)`, i.e. the unflattened action space of
                the current environment.
            * rewards(np.array): Array of rewards of shape :math:`(T,)`, i.e. a
                1D array of length timesteps.
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(np.array): Array of termination signals.
    """
    env_steps = []
    agent_infos = []
    observations = []
    if animated:
        if not is_softgym:
            env.visualize()
            #force headless bullet environment like inverted pendulum to start the server with GUI
            if hasattr(env._env, 'isRender'):
                env._env.isRender = True 

    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0

    img_size_softgym = 720
    
    image_array = []    
    while episode_length < (max_episode_length or np.inf):
        if not is_softgym:
            img = env._env.render(mode='rgb_array')
            image_array.append(img)

        if pause_per_frame is not None:
            time.sleep(pause_per_frame)
        a, agent_info = agent.get_action(last_obs)
        if deterministic and 'mean' in agent_info:
            a = agent_info['mean']
        if not is_softgym:
            es = env.step(a)
        else:
            # print(env, env._env._env)
            obs, reward, done, info = env._env._env.step(a, record_continuous_video=True, img_size=img_size_softgym)
            image_array.extend(info['flex_env_recorded_frames'])
            es = {'action':a, 'reward':reward, 'env_info':info, 'observation':obs, 'terminal':done}

        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        episode_length += 1
    
        if not is_softgym:
            if es.last:
                break
            last_obs = es.observation
        else:
            if es['terminal']:       #is done
                break
            last_obs = es['observation']
        
    
    if video_name is not None:
        save_numpy_to_video_matplotlib(image_array, video_name, interval=pause_per_frame*1000)

    if not is_softgym:
        return dict(
            episode_infos=episode_infos,
            observations=np.array(observations),
            actions=np.array([es.action for es in env_steps]),
            rewards=np.array([es.reward for es in env_steps]),
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
            dones=np.array([es.terminal for es in env_steps]),
        )
    else:
        return dict(
            episode_infos=episode_infos,
            observations=np.array(observations),
            actions=np.array([es['action'] for es in env_steps]),
            rewards=np.array([es['reward'] for es in env_steps]),
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list([es['env_info'] for es in env_steps]),
            dones=np.array([es['terminal'] for es in env_steps]),
        )

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--snapshot_dir', type=str, default='')
    parser.add_argument('--use_softgym', type=int, default=0)

    args = parser.parse_args()

    if args.snapshot_dir == '':
        print('Require the path to the snapshot to visualize the policy.')
        return

    print('Loading policy from {0}...'.format(args.snapshot_dir))
    
    snapshotter = Snapshotter()
    data = snapshotter.load(args.snapshot_dir)
    policy = data['algo'].policy
    env = data['env']

    policy.cpu()

    if args.use_softgym:
        try:
            import pyflex
            #headless, render, width, height
            pyflex.init(0, True, env._env.camera_width, env._env.camera_height) 
            env.reset()
        except ImportError as e:
            raise error.DependencyNotInstalled("{}. (You need to first compile the python binding)".format(e))
            return

    # See what the trained policy can accomplish
    print('Starting to run a rollout...')
    path = rollout(env, policy, max_episode_length=500, animated=False, video_name='test', pause_per_frame=0.01, deterministic=True, is_softgym=args.use_softgym)
    
    return

if __name__ == '__main__':
    main()