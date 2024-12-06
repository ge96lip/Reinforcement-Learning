from tqdm import trange
import torch
from utils import running_average

def ddpg_training(episodes, env, agent, n_ep_running_average = 50  ): 
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    EPISODES = trange(episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset environment data
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0

        while not (done or truncated):
            # Select an action using the DDPG agent
            action = agent.select_action(state)
            
            # Take the action in the environment
            next_state, reward, done, truncated, _ = env.step(action)

            # Add experience to the replay buffer
            agent.replay_buffer.add((state, action, reward, next_state, done))

            # Train the agent if replay buffer has enough samples
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train()

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Update tqdm progress bar with fresh information
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    name = './weights/neural-network-2'
    torch.save(agent.actor, name + '-actor'+'.pth')
    torch.save(agent.critic, name + '-critic'+'.pth')
    return episode_reward_list, episode_number_of_steps

def ddpg_training1(episodes, env, agent, n_ep_running_average = 50): 
    EPISODES = trange(episodes, desc='Episode: ', leave=True)
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    for i in EPISODES:
        # Reset environment data
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0

        while not (done or truncated):
            # Select an action using the DDPG agent
            action = agent.select_action(state)
            
            # Take the action in the environment
            next_state, reward, done, truncated, _ = env.step(action)

            # Add experience to the replay buffer
            agent.replay_buffer.add((state, action, reward, next_state, done))

            # Train the agent if replay buffer has enough samples
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train()

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Update tqdm progress bar with fresh information
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        
    return episode_reward_list, episode_number_of_steps
        
def random_training(episodes, env, agent, n_ep_running_average = 50): 
    EPISODES = trange(episodes, desc='Episode: ', leave=True)
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    for i in EPISODES:
        # Reset enviroment data
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0
        while not (done or truncated):
            # Take a random action
            action = agent.forward(state)

            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)


        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        
    return episode_reward_list, episode_number_of_steps