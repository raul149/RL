import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 16
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # TODO: Add another linear layer for the critic
        #Entry the hidden layers, output the value.
        self.fc3 = torch.nn.Linear(self.hidden, 1)
        self.sigma = torch.nn.Parameter(torch.tensor([10.]))  # TODO: Implement learned variance (or copy from Ex5)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = self.sigma  # TODO: Implement (or copy from Ex5)

        # Critic part
        # TODO: Implement
        state_val = self.fc3(x)
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma
        # Implement or copy from Ex5
        action_dist=Normal(action_mean,sigma)
        # TODO: Return state value in addition to the distribution

        return action_dist , state_val


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []

        # TODO: Compute state values (NO NEED FOR THE DISTRIBUTION)
        action_distr, pred_value_states=self.policy.forward(states)
        nextaction_distribution, valueprediction_next_states=self.policy.forward(next_states)
        ##### COMPUTED using the forward function
        #Critic Loss: 
        valueprediction_next_states = (valueprediction_next_states).squeeze(-1)
        pred_value_states = (pred_value_states).squeeze(-1)
        valueprediction_next_states = valueprediction_next_states*(1-done)
        critic_loss=F.mse_loss(pred_value_states, rewards+self.gamma*valueprediction_next_states.detach())
        # Advantage estimates
        # TODO: Compute advantage estimates
        advantage=rewards+self.gamma*valueprediction_next_states-pred_value_states
        # TODO: Calculate actor loss (very similar to PG)
        actor_loss=(-action_probs * advantage.detach()).mean()

        # TODO: Compute the gradients of loss w.r.t. network parameters
        # Or copy from Ex5
        loss=critic_loss+actor_loss
        #print(loss)
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients
        # Or copy from Ex5
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network
        # Or copy from Ex5
        action_distribution, __ = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy
        # Or copy from Ex5
        if evaluation:
            action = action_distribution.mean()
        else:
            action = action_distribution.sample()

        # TODO: Calculate the log probability of the action
        # Or copy from Ex5
        act_log_prob = action_distribution.log_prob(action)
        
        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        # Now we need to store some more information than with PG
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
