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
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()

        # Input: (4, 128, 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # C1 output size: (16, 64, 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # C2 output size: (32, 32, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )
        # C3 output size: (64, 1024)

        self.head1 = nn.Linear(64*32*32, 1)

        self.apply(weights_init_)

    def forward(self, state):
        s = self.conv1(state)
        s = self.conv2(s)
        s = self.conv3(s)
        v = self.head1(s)

        return v


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()

        # Input: (4, 128, 128)
        # Q1 architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # C1 output size: (16, 64, 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # C2 output size: (32, 32, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )
        # C3 output size: (64, 1024)

        self.head1 = nn.Linear(64*32*32 + num_actions, 1)

        # Q2 architecture
        self.conv4 = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        self.head2 = nn.Linear(64*32*32 + num_actions, 1)


        self.apply(weights_init_)

    def forward(self, state, action):
        s1 = self.conv1(state)
        s1 = self.conv2(s1)
        s1 = self.conv3(s1)
        v1 = self.head1(torch.cat([s1, action], 1))

        s2 = self.conv4(state)
        s2 = self.conv5(s2)
        s2 = self.conv6(s2)
        v2 = self.head2(torch.cat([s2, action], 1))
        
        return v1, v2


class GaussianPolicy(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        self.linear1 = nn.Linear(64*32*32, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear1(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
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
