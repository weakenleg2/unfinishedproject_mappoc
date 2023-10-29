
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import init, get_clones
from torch.distributions import Normal,Categorical
import math
import gymnasium as gym
import numpy as np
# 总的来说这个网络可以计算q_value, option policy, action policy

# not pretty sure do we really need to track the name
class Dense3D2(nn.Module):
    def __init__(self, in_features, out_features, num_options, weight_init=None, bias=True):
        super(Dense3D2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_options = num_options
        # print(in_features, in_features.shape)

        if isinstance(num_options, torch.Tensor):
            num_options = num_options.item()

        if isinstance(in_features, torch.Tensor):
            in_features = in_features.item()

        if isinstance(out_features, torch.Tensor):
            out_features = out_features.item()

        self.weight = nn.Parameter(torch.Tensor(num_options, in_features, out_features))

        self.weight = nn.Parameter(torch.Tensor(num_options, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_options, out_features))
        else:
            self.register_parameter('bias', None)
        if weight_init is not None:
            weight_init(self.weight)
        else:
            nn.init.orthogonal_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, option):
        # Select the weights and bias for the given option
        w = self.weight[option]
        b = self.bias[option] if self.bias is not None else None
        # print(self.weight.shape, option.shape)

        return F.linear(x, w, b)


# Instead of having a single set of weights and biases for the layer, it 
# has multiple sets, one for each "option". This allows the network to have
#  different transformations based on the selected option.
class RunningMeanStd:
    # This is a simple running mean and standard deviation calculator
    def __init__(self, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = torch.tensor(1e-4)

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @property
    def std(self):
        return torch.sqrt(self.var)

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N,output_dim,use_orthogonal=True, use_ReLU=True):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc_out = init_(nn.Linear(hidden_size, output_dim))  # Added this line to control the output dimension

        # layers = []
        # layers.append(nn.Linear(input_dim, hidden_size))
        # layers.append(nn.Tanh())
        # for _ in range(num_layers - 1):
        #     layers.append(nn.Linear(hidden_dim, hidden_size))
        #     layers.append(nn.Tanh())
        # layers.append(nn.Linear(hidden_size, output_size))
        # self.model = nn.Sequential(*layers)


    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        x = self.fc_out(x)
        return x

class MLPBase(nn.Module):
    def __init__(self,  ac_space,obs_shape,hidden_size,num,
                 cat_self=True, attn_internal=False):

        super(MLPBase, self).__init__()
        self.gaussian_fixed_var=True
        recurrent = False
        # self._use_feature_normalization = args.use_feature_normalization
        # self._use_orthogonal = args.use_orthogonal
        # self._use_ReLU = args.use_ReLU
        # self._stacked_frames = args.stacked_frames
        num_options=2
        dc=0, 
        hid_size =  1
        num_hid_layers= hidden_size
        self.num=int(num)
        # self.last_action = torch.zeros(ac_space.shape)
        # self.last_action_init = torch.zeros(ac_space.shape)
        self.ac_space = ac_space
        print(self.ac_space)
        # multi_walker 的特殊形式？


        # self.ac_space_dim = ac_space_dim
        # self.q_space_dim = q_space_dim
        # self.pi_space_dim = pi_space_dim
        # self.mu_space_dim = mu_space_dim
        # self.pdtype = pdtype = make_pdtype(ac_space) # GaussianDiag

        self.ac_space_dim = ac_space.n
        q_space_dim = (17+(num-1)*(5+4),)
        pi_space_dim = (17+(num-1)*(5),)
        mu_space_dim = (6,)
        self.q_space_dim = (17+(num-1)*(5+4),)
        self.pi_space_dim = (17+(num-1)*(5),)
        self.mu_space_dim = (6,)
        self.num_options = num_options
        self.dc = dc
        # if self._use_feature_normalization:
        #     self.feature_norm = nn.LayerNorm(obs_shape[0])
        self.ob_rms_q = RunningMeanStd(shape=(q_space_dim))
        self.ob_rms_pi = RunningMeanStd(shape=(pi_space_dim))
        self.ob_rms_mu = RunningMeanStd(shape=(mu_space_dim))
        self.ob = RunningMeanStd(shape=(obs_shape))

        self.q_net0 = MLPLayer(obs_shape, hid_size, num_hid_layers, self.ac_space_dim)
        self.q_net1 = MLPLayer(obs_shape, hid_size, num_hid_layers, self.ac_space_dim)

        self.option_net0 = MLPLayer(obs_shape, hid_size, num_hid_layers, 1)
        self.option_net1 = MLPLayer(obs_shape, hid_size, num_hid_layers,1)

        self.pi_net = MLPLayer(obs_shape, hid_size, num_hid_layers, self.ac_space_dim)

    def forward(self, ob, option, stochastic=True):
        print(ob)
        # print(f"Type of ob: {ob.dtype}")
        # print(f"Shape of ob: {ob.shape}")

        self.ob.update(ob)
        # obz_q = torch.clamp((ob[:, 3:self.q_space_dim + 3] - self.ob_rms_q.mean) / self.ob_rms_q.std, -10.0, 10.0)
        # obz_pi = torch.clamp((ob[:, 3:self.pi_space_dim + 3] - self.ob_rms_pi.mean) / self.ob_rms_pi.std, -10.0, 10.0)
        # obz_mu = torch.clamp((ob[:, :self.mu_space_dim] - self.ob_rms_mu.mean) / self.ob_rms_mu.std, -10.0, 10.0)
        ob = torch.clamp((ob-self.ob.mean)/self.ob.std,-10.0,10.0)
        # print(f"obtype:{type(self.ob)}")
        # print(f"obtypemean:{type(self.ob.mean)}")
        # print(f"obtypestd:{type(self.ob.std)}")
        #### q_network here
        q_val0 = self.q_net0(ob)
        q_val1 = self.q_net1(ob)
        # print(q_val0,q_val1)
        vpred = q_val1
        if option is not None:
            vpred = q_val1[:,0] if option[0] == 1 else q_val0[:,0]
        #####
        ### Define the policy over options
        option_val0 = self.option_net0(ob)
        option_val1 = self.option_net1(ob)
        if len(option_val0.shape) == 1:
            option_val0 = option_val0.unsqueeze(1)
        if len(option_val1.shape) == 1:
            option_val1 = option_val1.unsqueeze(1)

        option_vals = torch.cat([option_val0, option_val1], dim=1)
        op_pi = F.softmax(option_vals, dim=1)
        self.termhead = Dense3D2(option_vals.size(1), 1, num_options=self.num_options, weight_init=nn.init.orthogonal_)
        if option is not None:
            tpred_logits = self.termhead(option_vals, option)
            self.tpred = torch.sigmoid(tpred_logits).squeeze(-1)

        termination_sample = torch.tensor([True], dtype=torch.bool)
        ###
        ### Implementing intra-option-policy

        mean = self.pi_net(ob)
        # print(isinstance(self.ac_space, gym.spaces.Discrete))
        if self.ac_space.__class__.__name__ == "Box":
            self.logstd = nn.Parameter(torch.zeros(1, self.ac_space.shape[0]))

            std = torch.exp(self.logstd)
            self.pd = Normal(mean,std)
        elif self.ac_space.__class__.__name__ == "Discrete":
            self.logstd = torch.full((1, self.ac_space.n), -0.7)

            self.pd = Categorical(logits=mean)
            # 注意这个
        else:
            raise NotImplementedError
        # std = torch.exp(self.logstd)
        
        ac = self.pd.sample() if stochastic else self.pd.mode()
        # ac = self.pd.sample() if stochastic else mean

        ac = torch.clamp(ac, -1.0, 1.0)
        logstd = self.pd.log_prob(ac)


        return ac, vpred, logstd, op_pi
    # 不懂第三个为什么要返回那个， 是个obs clamp感觉没有必要返回(之前有个obz_pi)

    def act(self, stochastic, ob, option):
        ac, vpred, logstd,_ = self.forward(ob, option, stochastic)
        # return ac[0].detach().cpu().numpy(), vpred[0].detach().cpu().numpy(), logstd[0].detach().cpu().numpy()
        return ac.detach(), vpred.detach(), logstd.detach()


    def get_option(self, ob):
        op_prob = self.forward(ob, option = None)[3].detach().cpu().numpy()[0][0] 
        return np.random.choice(len(op_prob), p=op_prob)

    def get_term_adv(self, ob, curr_opt):
        vals = [self.forward(ob, torch.tensor([opt]))[1].detach().cpu().numpy()[0] for opt in range(self.num_options)]
        op_prob = self.forward(ob, option = None)[3].detach().cpu().numpy()[0]
        return (vals[curr_opt[0]] - np.sum(op_prob * vals) + self.dc), (vals[curr_opt[0]] - np.sum(op_prob * vals))

    def get_opt_adv(self, ob, curr_opt):
        vals = [self.forward(ob, torch.tensor([opt]))[1].detach().cpu().numpy()[0] for opt in range(self.num_options)]
        vals_max = np.max(vals)
        return (vals[curr_opt[0]] - vals_max + self.dc), (vals[curr_opt[0]] - vals_max)

    def get_opt_adv_oldpi(self, ob, curr_opt):
        # Same as get_opt_adv, not sure what you want to do with oldpi here. Adjust accordingly.
        vals = [self.forward(ob, torch.tensor([opt]))[1].detach().cpu().numpy()[0] for opt in range(self.num_options)]
        vals_max = np.max(vals)
        return (vals[curr_opt[0]] - vals_max + self.dc), (vals[curr_opt[0]] - vals_max)

    def get_variables(self):
        return list(self.parameters())

    def get_trainable_variables(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))

    def get_initial_state(self):
        return []  

    def reset_last_act(self):
        self.last_action = self.last_action_init.detach()
        return self.last_action