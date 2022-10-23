from ddpg_agent import Agent, ReplayBuffer

import torch


class MADDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed, buffer_size, batch_size, gamma, tau, lr, weight_decay ):
        """Initializes multiple DDPG agents.
        Params:
        ======
            num_agents (int): number of agents in the environment
            state_size (int): dimension of each state per agent
            action_size (int): dimension of each action per agent
            random_seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate of the actor and of the critic
            weight_decay (float): L2 weight decay
        """
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.gamma = gamma

        #Create N number of agents
        self.agents = [Agent(state_size, action_size, random_seed, buffer_size, batch_size, gamma, tau, lr, weight_decay) for _ in range(self.num_agents)]

        #Create shared replay buffer for all agents
        self.memory = ReplayBuffer(action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn for each agent in the environment"""
        # Save experience / reward
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if(len(self.memory) > self.batch_size):
            for _ in range(self.num_agents):
                experience = self.memory.sample()
                self.learn(experience)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy for each agent in the environment"""
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state, add_noise)
            actions.append(action)
        return actions
    
    def reset(self):
        """Reset the noise value for each agent in the environment"""
        for agent in self.agents:
            agent.reset()
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples, for each agents simultaneously"""
        for agent in self.agents:
            agent.learn(experiences, self.gamma)