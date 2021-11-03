import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=1.0, std=1.0)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, input_space: dict):
        super(ValueNetwork, self).__init__()

        _map_space = input_space['map'].shape
        _lidar_space = input_space['lidar'].shape
        _goal_space = input_space['goal'].shape

        # map's feature
        self.conv1 = nn.Sequential(
            nn.Conv2d(_map_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        # lidar's feature
        self.c1 = nn.Sequential(
            nn.Conv1d(_lidar_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(8*90, 256)
        )

        self.g = nn.Linear(3, 8)

        self.head1 = nn.Linear(16*20*20 + 256 + 8, 1) # map+lidar+goal
        # self.head1 = nn.Linear(64*32*32, 1)

        self.apply(weights_init_)

    def forward(self, state: dict):
        s = self.conv1(state['map'])
        l = self.c1(state['lidar'])
        g = self.g(state['goal'])
        g = g.squeeze()
        # print("s shape {}".format(s.shape)) # Size([batch_size, 65536])
        # print("l shape {}".format(l.shape)) # Size([batch_size, 270])
        # print("g shape {}".format(g.shape)) # Size([batch_size, 3])
        v = self.head1(torch.cat([s, l, g], 1))

        return v


class QNetwork(nn.Module):
    def __init__(self, input_space: dict, num_actions):
        super(QNetwork, self).__init__()

        _map_space = input_space['map'].shape
        _lidar_space = input_space['lidar'].shape
        _goal_space = input_space['goal'].shape

        # Q1 architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(_map_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        # lidar's feature
        self.c1 = nn.Sequential(
            nn.Conv1d(_lidar_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(8*90, 256)
        )

        self.g1 = nn.Linear(3, 8)

        self.head1 = nn.Linear(16*20*20 + 256 + 8 + num_actions, 1) # O(map+lidar+goal)+A(3)

        # Q2 architecture
        self.conv2 = nn.Sequential(
            nn.Conv2d(_map_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(_lidar_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(8*90, 256)
        )

        self.g2 = nn.Linear(3, 8)

        self.head2 = nn.Linear(16*20*20+ 256 + 8 + num_actions, 1) # O(map+lidar+goal)+A(3)

        self.apply(weights_init_)

    def forward(self, state: dict, action):
        s1 = self.conv1(state['map'])
        s1 = s1.squeeze()
        l1 = self.c1(state['lidar'])
        l1 = l1.squeeze()
        g1 = self.g1(state['goal'])
        g1 = g1.squeeze()
        # print("s1 shape {}".format(s1.shape)) # Size([batch_size, 65536])
        # print("l1 shape {}".format(l1.shape)) # Size([batch_size, 256])
        # print("g1 shape {}".format(g1.shape)) # Size([batch_size, 8])
        # print("a shape {}".format(action.shape)) # Size([batch_size, 3])
        # print("cat shape {}".format(torch.cat([s1, l1, g1, action], 1).shape)) # Size([batch_size, 65536+256+8+3])
        v1 = self.head1(torch.cat([s1, l1, g1, action], 1))

        s2 = self.conv2(state['map'])
        s2 = s2.squeeze()
        l2 = self.c2(state['lidar'])
        l2 = l2.squeeze()
        g2 = self.g2(state['goal'])
        g2 = g2.squeeze()
        v2 = self.head2(torch.cat([s2, l2, g2, action], 1))
        
        return v1, v2


class GaussianPolicy(nn.Module):
    def __init__(self, input_space: dict, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        _map_space = input_space['map'].shape
        _lidar_space = input_space['lidar'].shape
        _goal_space = input_space['goal'].shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(_map_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        # lidar's feature
        # Input: (1, 270)
        self.c1 = nn.Sequential(
            nn.Conv1d(_lidar_space[0], 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(8*90, 256)
        )

        self.g = nn.Linear(3, 8)

        self.linear1 = nn.Linear(16*20*20 + 256 + 8, hidden_dim) # map_feature+lidar_feature+goal_feature

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state: dict):
        x = self.conv1(state['map'])
        l = self.c1(state['lidar'])
        # print("g shape {}".format(state['goal'].shape)) # Size([batch_size, 8]) # Size([8])
        g = self.g(state['goal'].squeeze())
        if g.shape == torch.Size([8]): # I know it's weird...
            g = g.unsqueeze(0)
        # print("x shape {}".format(x.shape)) # Size([batch_size, 65536]) # Size([1, 65536])
        # print("l shape {}".format(l.shape)) # Size([batch_size, 270]) # Size([1, 270])
        # print("g shape {}".format(g.shape)) # Size([batch_size, 8]) # Size([8])
        x = self.linear1(torch.cat([x, l, g], 1))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: dict):
        # mean, log_std = self.forward(state)
        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # return action, log_prob, mean, log_std
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

# class DeterministicPolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(DeterministicPolicy, self).__init__()
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)

#         self.mean = nn.Linear(hidden_dim, num_actions)
#         self.noise = torch.Tensor(num_actions)

#         self.apply(weights_init_)

#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
#         return mean

#     def sample(self, state):
#         mean = self.forward(state)
#         noise = self.noise.normal_(0., std=0.1)
#         noise = noise.clamp(-0.25, 0.25)
#         action = mean + noise
#         return action, torch.tensor(0.), mean
