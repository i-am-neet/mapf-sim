#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical



################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:0')
# 
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
    
print("============================================================================================")



def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Actor(nn.Module):
    def __init__(self, input_space, num_actions, has_continuous_action_space, action_std_init, action_space):
        super(Actor, self).__init__()

        hidden_dim = 256
        _map_space = input_space['map'].shape
        _lidar_space = input_space['lidar'].shape
        _goal_space = input_space['goal'].shape
        _plan_len_space = input_space['plan_len'].shape
        _robot_info_space = input_space['robot_info'].shape

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = num_actions
            self.action_var = torch.full((num_actions,), action_std_init * action_std_init).to(device)

        self.conv1 = nn.Sequential(
            nn.Conv2d(_map_space[0], 16, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(), # or batch_normalize
            nn.Flatten(),
            nn.SELU(),
            nn.Linear(32*20*20, hidden_dim//2),
            nn.SELU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SELU()
        )

        # lidar's feature
        self.l1 = nn.Sequential(
            nn.Linear(_lidar_space[0], hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SELU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SELU()
        )

        ## MindDa Structure
        self.info = nn.Sequential(
            nn.Linear(_robot_info_space[0] + _goal_space[0] + _plan_len_space[0] + hidden_dim//4, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim//4 + hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SELU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )

        self.mean_linear = nn.Linear(hidden_dim//4, num_actions)

        self.apply(weights_init_)
        
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self, state: dict):
        x = self.conv1(state['map'])
        l = self.l1(state['lidar'])
        i = self.info(torch.cat([state['robot_info'], state['goal'], state['plan_len'], l], len(l.shape)-1))
        x = x.unsqueeze(1).squeeze()
        i = i.squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            i = i.unsqueeze(0)
        x = self.fc1(torch.cat([x, i], len(i.shape)-1))
        mean = self.mean_linear(x)
        return mean

    def sample(self, state: dict):
        action_mean = self.forward(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        a = torch.tanh(action) * self.action_scale + self.action_bias

        return a.detach(), action_logprob.detach()

class Critic(nn.Module):
    def __init__(self, input_space, num_actions, has_continuous_action_space, action_std_init):
        super(Critic, self).__init__()

        hidden_dim = 256
        _map_space = input_space['map'].shape
        _lidar_space = input_space['lidar'].shape
        _goal_space = input_space['goal'].shape
        _plan_len_space = input_space['plan_len'].shape
        _robot_info_space = input_space['robot_info'].shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(_map_space[0], 16, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(), # or batch_normalize
            nn.Flatten(),
            nn.SELU(),
            nn.Linear(32*20*20, hidden_dim//2),
            nn.SELU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SELU()
        )

        self.l1 = nn.Sequential(
            nn.Linear(_lidar_space[0], hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SELU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SELU()
        )

        self.info1 = nn.Sequential(
            nn.Linear(_robot_info_space[0] + _goal_space[0] + _plan_len_space[0] + hidden_dim//4, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        )

        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim//4 + hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SELU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SELU(),
            nn.Linear(hidden_dim//4, 1)
        )

        self.apply(weights_init_)

    def forward(self, state: dict):
        s1 = self.conv1(state['map'])
        s1 = s1.squeeze()
        l1 = self.l1(state['lidar'])
        i1 = self.info1(torch.cat([state['robot_info'], state['goal'], state['plan_len'], l1], 1))
        i1 = i1.squeeze()
        v1 = self.head1(torch.cat([s1, i1], 1))
        
        return v1

class PPO:
    def __init__(self, state_space, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, action_space=None):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = Actor(state_space, action_dim, has_continuous_action_space, action_std_init, action_space).to(device)
        self.critic = Critic(state_space, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy_old = Actor(state_space, action_dim, has_continuous_action_space, action_std_init, action_space).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.policy.forward(state)
            
            action_var = self.policy.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic.forward(state)
        
        return action_logprobs, state_values, dist_entropy


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state: dict):

        state_map = torch.FloatTensor(state['map']).to(device).unsqueeze(0)
        state_lidar = torch.FloatTensor(state['lidar']).to(device).unsqueeze(0)
        state_goal = torch.FloatTensor(state['goal']).to(device).unsqueeze(0)
        state_plan_len = torch.FloatTensor(state['plan_len']).to(device).unsqueeze(0)
        state_robot_info= torch.FloatTensor(state['robot_info']).to(device).unsqueeze(0)
        state = {'map': state_map, 'lidar': state_lidar, 'goal': state_goal, 'plan_len': state_plan_len, 'robot_info': state_robot_info}

        if self.has_continuous_action_space:
            with torch.no_grad():
                action, action_logprob = self.policy_old.sample(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.sample(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = {'map':[], 'lidar':[], 'goal':[], 'plan_len':[], 'robot_info':[]}
        for s in self.buffer.states:
            old_states['map'].append(s['map'])
            old_states['lidar'].append(s['lidar'])
            old_states['goal'].append(s['goal'])
            old_states['plan_len'].append(s['plan_len'])
            old_states['robot_info'].append(s['robot_info'])

        old_states['map'] = torch.squeeze(torch.stack(old_states['map'], dim=0)).unsqueeze(1)
        old_states['lidar'] = torch.squeeze(torch.stack(old_states['lidar'], dim=0))
        old_states['goal'] = torch.squeeze(torch.stack(old_states['goal'], dim=0))
        old_states['plan_len'] = torch.squeeze(torch.stack(old_states['plan_len'], dim=0)).unsqueeze(1)
        old_states['robot_info'] = torch.squeeze(torch.stack(old_states['robot_info'], dim=0))

        # convert list to tensor
        # old_state['map'] = torch.FloatTensor(old_state['map']).detach().to(self.device)
        # old_state['lidar'] = torch.FloatTensor(old_state['lidar']).detach().to(self.device)
        # old_state['goal'] = torch.FloatTensor(old_state['goal']).detach().to(self.device)
        # old_state['plan_len'] = torch.FloatTensor(old_state['plan_len']).detach().to(self.device)
        # old_state['robot_info'] = torch.FloatTensor(old_state['robot_info']).detach().to(self.device)
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        losses = [] # for writer
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            losses.append(loss)

            # take gradient step
            self.policy_optim.zero_grad()
            loss.mean().backward()
            self.policy_optim.step()

            # self.critic_optim.zero_grad()
            # critic_loss.backward()
            # self.critic_optim.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return losses
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


