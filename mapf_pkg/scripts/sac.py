import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
# SAC_V
# from model import GaussianPolicy, QNetwork, ValueNetwork
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, input_space, action_space, args):

        self.use_expert = args.use_expert
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_range = [action_space.low, action_space.high]
        self.policy_type = args.policy

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        # self.device = torch.device("cuda" if args.cuda else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(torch.cuda.is_available())
        # print(torch.cuda.current_device())
        # print(torch.cuda.device(0))
        # print(torch.cuda.device_count())
        # print(torch.cuda.get_device_name())
        # print(torch.backends.cudnn.version())
        # print(torch.backends.cudnn.is_available())

        self.critic = QNetwork(input_space, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(input_space, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(input_space, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(input_space, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # SAC_V
        # self.value = ValueNetwork(input_space).to(device=self.device)
        # self.value_target = ValueNetwork(input_space).to(self.device)
        # self.value_optim = Adam(self.value.parameters(), lr=args.lr)
        # hard_update(self.value_target, self.value)

        # self.policy = GaussianPolicy(input_space, action_space.shape[0], args.hidden_size).to(self.device)
        # self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state_map = torch.FloatTensor(state['map']).to(self.device).unsqueeze(0)
        state_lidar = torch.FloatTensor(state['lidar']).to(self.device).unsqueeze(0)
        state_goal = torch.FloatTensor(state['goal']).to(self.device).unsqueeze(0)
        state_plan_len = torch.FloatTensor(state['plan_len']).to(self.device).unsqueeze(0)
        state = {'map': state_map, 'lidar': state_lidar, 'goal': state_goal, 'plan_len': state_plan_len}
        if eval is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]
        # return self.rescale_action(action)
        return action

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
                (self.action_range[1] + self.action_range[0]) / 2.0

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        if not self.use_expert:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, s_e_batch, a_e_batch = memory.sample(batch_size=batch_size, use_expert=True)

        # State is array of dictionary like [{"map":value, "lidar":value, "goal":value}, ...]
        # So, convert list to dict below:
        _state_batch = {'map':[], 'lidar':[], 'goal':[], 'plan_len':[]}
        _next_state_batch = {'map':[], 'lidar':[], 'goal':[], 'plan_len':[]}
        _state_expert_batch = {'map':[], 'lidar':[], 'goal':[], 'plan_len':[]}
        for s in state_batch:
            _state_batch['map'].append(s['map'])
            _state_batch['lidar'].append(s['lidar'])
            _state_batch['goal'].append(s['goal'])
            _state_batch['plan_len'].append(s['plan_len'])

        for s in next_state_batch:
            _next_state_batch['map'].append(s['map'])
            _next_state_batch['lidar'].append(s['lidar'])
            _next_state_batch['goal'].append(s['goal'])
            _next_state_batch['plan_len'].append(s['plan_len'])

        if self.use_expert:
            for s in s_e_batch:
                _state_expert_batch['map'].append(s['map'])
                _state_expert_batch['lidar'].append(s['lidar'])
                _state_expert_batch['goal'].append(s['goal'])
                _state_expert_batch['plan_len'].append(s['plan_len'])

        _state_batch['map'] = torch.FloatTensor(_state_batch['map']).to(self.device)
        _state_batch['lidar'] = torch.FloatTensor(_state_batch['lidar']).to(self.device)
        _state_batch['goal'] = torch.FloatTensor(_state_batch['goal']).to(self.device)
        _state_batch['plan_len'] = torch.FloatTensor(_state_batch['plan_len']).to(self.device)
        _next_state_batch['map'] = torch.FloatTensor(_next_state_batch['map']).to(self.device)
        _next_state_batch['lidar'] = torch.FloatTensor(_next_state_batch['lidar']).to(self.device)
        _next_state_batch['goal'] = torch.FloatTensor(_next_state_batch['goal']).to(self.device)
        _next_state_batch['plan_len'] = torch.FloatTensor(_next_state_batch['plan_len']).to(self.device)
        if self.use_expert:
            _state_expert_batch['map'] = torch.FloatTensor(_state_expert_batch['map']).to(self.device)
            _state_expert_batch['lidar'] = torch.FloatTensor(_state_expert_batch['lidar']).to(self.device)
            _state_expert_batch['goal'] = torch.FloatTensor(_state_expert_batch['goal']).to(self.device)
            _state_expert_batch['plan_len'] = torch.FloatTensor(_state_expert_batch['plan_len']).to(self.device)
            _action_expert_batch = torch.FloatTensor(a_e_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # SAC_V
        # with torch.no_grad():
        #     vf_next_target = self.value_target(_new_next_state_batch)
        #     next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(_next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(_next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(_state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update Policy
        if not self.use_expert:
            pi, log_pi, _ = self.policy.sample(_state_batch)

            qf1_pi, qf2_pi = self.critic(_state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
        else:
            pi, log_pi, _ = self.policy.sample(_state_expert_batch)

            qf1_pi, qf2_pi = self.critic(_state_expert_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

        # # SAC_V
        # # Regularization Loss
        # reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        # policy_loss += reg_loss

        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        # self.policy_optim.step()

        # # Update Value
        # if not self.use_expert:
        #     vf = self.value(_new_state_batch)
        # else:
        #     vf = self.value(_new_s_e_batch)
        
        # with torch.no_grad():
        #     vf_target = min_qf_pi - (self.alpha * log_pi)

        # vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

        # self.value_optim.zero_grad()
        # vf_loss.backward()
        # self.value_optim.step()

        # if updates % self.target_update_interval == 0:
        #     soft_update(self.value_target, self.value, self.tau)

        # return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to\n {}\n, {}\n, {}\n, {}\n, {}\n and {}'.format(actor_path, critic_path))

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from\n {}\n, {}\n, {}\n, {}\n, {}\n and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
