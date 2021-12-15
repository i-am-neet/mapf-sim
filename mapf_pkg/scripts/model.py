import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = -0.5
LOG_SIG_MIN = -3
epsilon = 1e-6

# Initialize Policy weights
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

class QNetwork(nn.Module):
    def __init__(self, input_space: dict, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        _map_space = input_space['map'].shape
        _lidar_space = input_space['lidar'].shape
        _goal_space = input_space['goal'].shape
        _plan_len_space = input_space['plan_len'].shape
        _robot_info_space = input_space['robot_info'].shape

        # Q1 architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(_map_space[0], 6, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(6),
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(), # or batch_normalize
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(120*10*10, hidden_dim),
            # nn.SELU(),
            # nn.Linear(hidden_dim//2, hidden_dim//4),
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
        self.info1 = nn.Sequential(
            nn.Linear(_robot_info_space[0] + _goal_space[0] + _plan_len_space[0] + hidden_dim//4, hidden_dim),
            nn.SELU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        )

        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + num_actions, hidden_dim),
            # nn.Linear(16*20*20+ 256 + 8 + num_actions, hidden_dim), # O(map+lidar+goal)+A(3)
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.conv2 = nn.Sequential(
            nn.Conv2d(_map_space[0], 6, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(6),
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(120*10*10, hidden_dim),
            nn.SELU()
        )

        self.l2 = nn.Sequential(
            nn.Linear(_lidar_space[0], hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SELU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SELU()
        )

        ## MindDa Structure
        self.info2 = nn.Sequential(
            nn.Linear(_robot_info_space[0] + _goal_space[0] + _plan_len_space[0] + hidden_dim//4, hidden_dim),
            nn.SELU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        )

        self.head2 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + num_actions, hidden_dim), # O(map+lidar+goal+plan)+A(3)
            # nn.Linear(16*20*20+ 256 + 8 + num_actions, hidden_dim), # O(map+lidar+goal)+A(3)
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init_)

    def forward(self, state: dict, action):
        action = action.squeeze()

        s1 = self.conv1(state['map'])
        s1 = s1.squeeze()
        l1 = self.l1(state['lidar'])
        i1 = self.info1(torch.cat([state['robot_info'], state['goal'], state['plan_len'], l1], 2))
        i1 = i1.squeeze()
        v1 = self.head1(torch.cat([s1, i1, action], 1))
        # g1 = self.g1(state['goal'])
        # g1 = g1.squeeze()
        # p1 = self.p1(state['plan_len'].unsqueeze(1))
        # p1 = p1.squeeze()
        # v1 = self.head1(torch.cat([s1, l1, g1, p1, action], 1))
        # v1 = self.head1(torch.cat([s1, l1, g1, action], 1))

        s2 = self.conv2(state['map'])
        s2 = s2.squeeze()
        l2 = self.l2(state['lidar'])
        # g2 = self.g2(state['goal'])
        # g2 = g2.squeeze()
        # p2 = self.p2(state['plan_len'].unsqueeze(1))
        # p2 = p2.squeeze()
        i2 = self.info2(torch.cat([state['robot_info'], state['goal'], state['plan_len'], l2], 2))
        i2 = i2.squeeze()
        v2 = self.head2(torch.cat([s2, i2, action], 1))
        # v2 = self.head2(torch.cat([s2, l2, g2, p2, action], 1))
        # v2 = self.head2(torch.cat([s2, l2, g2, action], 1))
        
        return v1, v2


class GaussianPolicy(nn.Module):
    def __init__(self, input_space: dict, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        _map_space = input_space['map'].shape
        _lidar_space = input_space['lidar'].shape
        _goal_space = input_space['goal'].shape
        _plan_len_space = input_space['plan_len'].shape
        _robot_info_space = input_space['robot_info'].shape

        self.conv1 = nn.Sequential(
            # nn.Conv2d(_map_space[0], 16, kernel_size=3, stride=1, padding=1),
            # nn.SELU(),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.SELU(),
            # nn.MaxPool2d(2),
            # nn.Flatten(),
            # nn.SELU(),
            # nn.Linear(32*20*20, hidden_dim//2),
            # nn.SELU(),
            # nn.Linear(hidden_dim//2, hidden_dim//4),
            # nn.SELU()
            nn.Conv2d(_map_space[0], 6, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(6),
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(120*10*10, hidden_dim),
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
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            # nn.Linear(hidden_dim//4 + hidden_dim//4 + 8 + 8, hidden_dim), # map_feature+lidar_feature+goal_feature
            # nn.Linear(16*20*20 + 256 + 8, hidden_dim), # map_feature+lidar_feature+goal_feature
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
            # nn.SELU(),
            # nn.Linear(hidden_dim//2, hidden_dim//4)
        )

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state: dict):
        x = self.conv1(state['map'])
        l = self.l1(state['lidar'])
        # g = self.g(state['goal'].squeeze())
        # if state['plan_len'].squeeze().shape == torch.Size([]):
        #     p = self.p(state['plan_len'].squeeze().unsqueeze(0))
        # else:
        #     p = self.p(state['plan_len'].squeeze().unsqueeze(1))
        # if g.shape == torch.Size([8]): # I know it's weird...
        #     g = g.unsqueeze(0)
        # if p.shape == torch.Size([8]): # I know it's weird...
        #     p = p.unsqueeze(0)
        # if l.shape == torch.Size([64]): # I know it's weird...
        #     l = l.unsqueeze(0)
        i = self.info(torch.cat([state['robot_info'], state['goal'], state['plan_len'], l], len(l.shape)-1))
        x = x.unsqueeze(1).squeeze()
        i = i.squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            i = i.unsqueeze(0)
        x = self.fc1(torch.cat([x, i], len(i.shape)-1))
        # x = self.fc1(torch.cat([x, l, g, p], 1))
        # x = self.linear1(torch.cat([x, l, g], 1))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: dict):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
        # SAC_V
        # mean, log_std = self.forward(state)
        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action = torch.tanh(x_t)
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(1 - action.pow(2) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        # return action, log_prob, mean, log_std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
