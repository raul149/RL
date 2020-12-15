import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.episode = 0 #Task 2: Instead of modifying the cartpole.py (A hacky solution)
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        #Sigma for task 1
        #self.sigma = torch.tensor([5.])
        #Sigma for task 2a
        self.sigma = torch.tensor([10.])
        #Sigma for Task 2b
        #self.sigma = torch.nn.Parameter(torch.tensor([10.]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        #Sigma for Task 1 and task 2b
        #sigma=self.sigma
        #Sigma for task 2a (Comment for other tasks)
        sigma = self.sigma * np.exp((-5e-4)*self.episode)
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)
        action_distribution=Normal(action_mean,sigma)
        return action_distribution


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []
        self.policy.episode = episode_number #To store the value of the episode,used for 2a
        #Baseline value UNCOMMENT FOR TASK 1B
        #baseline=20
        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        #Normalized, comment for task 1a,1b. Uncomment for the rest
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        
        # TODO: Compute the optimization term (T1)
        #Task 1a/1c/2
        weighted_probs = action_probs * discounted_rewards
        #Task 1b (Uncomment for task 1b)
        #weighted_probs = action_probs * (discounted_rewards - baseline)
        #Common for all tasks
        loss = torch.sum(-weighted_probs) #We could use as well the torch.mean, we use negative as the gradient looks to maximize
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        # TODO: Pass state x through the policy network (T1)
        action_distribution = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = action_distribution.mean
        else:
            action = action_distribution.sample()
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = action_distribution.log_prob(action)

        return action, act_log_prob



    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

