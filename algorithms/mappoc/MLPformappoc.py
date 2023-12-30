
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.mappoc.utils.util import init, get_clones
from torch.distributions import Normal,Categorical
import math
import gymnasium as gym
import numpy as np
# 总的来说这个网络可以计算q_value, option policy, action policy

# not pretty sure do we really need to track the name
def prepare_observation(obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    if obs_tensor.ndim == 1:
        # print("true")
        # 单个观测，增加一个批次维度
        obs_tensor = obs_tensor.unsqueeze(0)
    # 如果已经是批量观测，则不需要改动
    return obs_tensor

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
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_options, out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        if weight_init is not None:
            weight_init(self.weight)
        else:
            nn.init.orthogonal_(self.weight, a=math.sqrt(5))
       

    def forward(self, x, option):
        # Select the weights and bias for the given option
        # print(option)
        # print(self.weight.shape, option.shape)
        if not isinstance(option, int):
            option = option.long().item()
        w = self.weight[option]
        # print(f"w:{w}")
        # print(f"bias:{self.bias}")

        b = self.bias[option] if self.bias is not None else None
        


        return F.linear(x, w.T, b)


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
        # 应该是0吧
        # print(f"x:{x}")
        batch_mean = torch.mean(x, dim=0)
        # print(f"mean:{batch_mean}")
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
        # print(f"input_dim:{input_dim}")
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc_out = init_(nn.Linear(hidden_size, output_dim))  # Added this line to control the output dimension


    def forward(self, x):
        # print("Shape of input (mat1):", x.shape)
        x = self.fc1(x)
        # print("Shape of input (mat1):", x.shape)
        # print("Shape of weight matrix (mat2):", self.fc1[0].weight.shape)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        x = self.fc_out(x)
        return x

class MLPBase(nn.Module):
    def __init__(self, hid_size,num_hid_layers,num_options,dc,q_space,ac_space,pi_space, mu_space,num,
                 gaussian_fixed_var=True):

        super(MLPBase, self).__init__()
        self.gaussian_fixed_var=gaussian_fixed_var
        self.recurrent = False
        num_options=num_options
        dc=dc, 
        hid_size =  hid_size
        num_hid_layers= num_hid_layers
        self.num=int(num)

        self.ac_space = ac_space

        self.ac_space_dim = ac_space.shape[0]
        # print(f"ac_space.shape[0]:{ac_space.shape[0]}")
        self.q_space_dim = q_space[0]
        self.pi_space_dim = pi_space[0]
        self.mu_space_dim = mu_space[0]
        # print(self.ac_space_dim,self.q_space_dim,self.pi_space_dim,self.mu_space_dim)
        # print(f"self.mu:{self.mu_space_dim}")
        # self.pdtype = DiagonalGaussianPolicy(ac_space)
        # print(self.ac_space)
        # multi_walker 的特殊形式？
        self.num_options = num_options
        # print(num_options)
        self.dc = dc
        # if self._use_feature_normalization:
        #     self.feature_norm = nn.LayerNorm(obs_shape[0])
        self.ob_rms_q = RunningMeanStd(shape=(self.q_space_dim))
        self.ob_rms_pi = RunningMeanStd(shape=(self.pi_space_dim))
        self.ob_rms_mu = RunningMeanStd(shape=(self.mu_space_dim))
        # self.ob = RunningMeanStd(shape=(obs_shape))

        self.q_net0 = MLPLayer(self.q_space_dim, hid_size, num_hid_layers, 1)
        self.q_net1 = MLPLayer(self.q_space_dim, hid_size, num_hid_layers, 1)

        self.option_net0 = MLPLayer(self.mu_space_dim, hid_size, num_hid_layers, 1)
        self.option_net1 = MLPLayer(self.mu_space_dim, hid_size, num_hid_layers,1)

        self.pi_net = MLPLayer(self.pi_space_dim, hid_size, num_hid_layers, self.ac_space_dim)

    def forward(self, ob, option, stochastic=True):
        # print(ob)
        # print(f"Type of ob: {ob.dtype}")
        # print(f"Shape of ob: {ob.shape}")

        # self.ob.update(ob)
        # self.ob_rms_q.update()
        # print(f"ob:{ob}")
        ob = prepare_observation(ob)
        print(f"ob.shape[0]:{ob.shape}")
        if option is not None:
            option = torch.tensor(option, dtype=torch.long).unsqueeze(0)
        # print(f"option:{option}")
        # print(f"(ob[:, 3:self.q_space_dim + 3]:{ob[:, 3:self.q_space_dim + 3].shape}")
        # print(self.ob_rms_q.mean.shape)
        

        # print(f"torch.mean(ob[:, 3:self.q_space_dim + 3],dim=0):{torch.mean(ob[:, 3:self.q_space_dim + 3],dim=0)}")
        # print(ob[:, 3:self.q_space_dim + 3] - torch.mean(ob[:, 3:self.q_space_dim + 3],dim=0))
        if ob.shape[0]==1:
            obz_q = torch.clamp((ob[:, 3:self.q_space_dim + 3]), -10.0, 10.0)
            obz_pi = torch.clamp((ob[:, 3:self.pi_space_dim + 3]), -10.0, 10.0)
            obz_mu = torch.clamp((ob[:, :self.mu_space_dim]), -10.0, 10.0)
            # print("pass")
        else:
            self.ob_rms_q.update(ob[:, 3:self.q_space_dim + 3])
            self.ob_rms_pi.update(ob[:, 3:self.pi_space_dim + 3])
            self.ob_rms_mu.update(ob[:, self.mu_space_dim])
            obz_q = torch.clamp((ob[:, 3:self.q_space_dim + 3] - self.ob_rms_q.mean) / self.ob_rms_q.std, -10.0, 10.0)
            obz_pi = torch.clamp((ob[:, 3:self.pi_space_dim + 3] - self.ob_rms_pi.mean) / self.ob_rms_pi.std, -10.0, 10.0)
            obz_mu = torch.clamp((ob[:, :self.mu_space_dim] - self.ob_rms_mu.mean) / self.ob_rms_mu.std, -10.0, 10.0)
        
        #### q_network here
        # print(f"obz_q:{obz_q}")
        q_val0 = self.q_net0(obz_q)
        q_val1 = self.q_net1(obz_q)
        # vpred = q_val1
        if option is not None:
            # 如果option是none,预计self.vpred不会改变，保持上次的值
            self.vpred = q_val1[:,0] if option[0] == 1 else q_val0[:,0]
        # self.vpred=vpred
        #####
        ### Define the policy over options
        # print(f"obz_mu:{obz_mu}")
        option_val0 = self.option_net0(obz_mu)
        # print(f"option_val0:{option_val0}")
        option_val1 = self.option_net1(obz_mu)
        if len(option_val0.shape) == 1:
            option_val0 = option_val0.unsqueeze(1)
        if len(option_val1.shape) == 1:
            option_val1 = option_val1.unsqueeze(1)
        # print(f"option_val01:{option_val0}")

        option_vals = torch.cat([option_val0, option_val1], dim=1)
        # print(option_vals)
        
        op_pi = F.softmax(option_vals,dim=1)
        # print(f"op_1:{op_pi}")
        # op_pi2 = F.softmax(option_vals)
        # print(f"op_2:{op_pi2}")

        self.op_pi = op_pi
        # print(f"option_vals:{option_vals.size(1)}")
        self.termhead = Dense3D2(option_vals.size(1), 1, num_options=self.num_options, weight_init=nn.init.orthogonal_)
        # print(option.shape)
        if option is not None:
            tpred_logits = self.termhead(option_vals.detach(), option)
            # print(f"tpred_logits:{tpred_logits}")
            self.tpred = torch.sigmoid(tpred_logits)[:,0]
            # print(self.tpred)
        # tpred检查完毕
        # print(f"tpred:{self.tpred}")
        self.termination_sample = torch.tensor([True], dtype=torch.bool)
        ###
        ### Implementing intra-option-policy

        mean = self.pi_net(obz_pi)
        # print(isinstance(self.ac_space, gym.spaces.Discrete))
        if self.ac_space.__class__.__name__ == "Box" and self.gaussian_fixed_var:
            self.logstd = nn.Parameter(torch.zeros(1, self.ac_space_dim))
            # print("BOX")

            std = torch.exp(self.logstd)
            self.pd = Normal(mean,std)
        else:
            self.logstd = torch.full((1, self.ac_space_dim), -0.7)
            print("discrete")

            self.pd = Categorical(logits=mean)
            # 注意这个
        
        # std = torch.exp(self.logstd)
        
        ac = self.pd.sample() if stochastic else self.pd.mode()
        # ac = self.pd.sample() if stochastic else mean

        ac = torch.clamp(ac, -1.0, 1.0)
        logstd = self.pd.log_prob(ac)


        return ac, self.vpred, mean, logstd, op_pi,self.tpred
    # 不懂第三个为什么要返回那个， 是个obs clamp感觉没有必要返回(之前有个obz_pi)

    def act(self, stochastic, ob, option):
        # print(option,type(option))
        ac, vpred, mean,logstd,_,_ = self.forward(ob, option, stochastic)
        # print(f"ac:{ac[0]}")
        # print(f"vpred:{vpred[0]}")
        # print(f"mean:{mean}")
        # print(f"logstd:{logstd[0]}")

        # return ac[0].detach().cpu().numpy(), vpred[0].detach().cpu().numpy(), logstd[0].detach().cpu().numpy()
        return ac[0].detach().cpu().numpy(), vpred[0].detach().cpu().numpy(), mean[0].detach().cpu().numpy(), logstd[0].detach().cpu().numpy()
    def get_vpred(self, ob, option):
        # print(option,type(option))
        ac, vpred, mean,logstd,_,_ = self.forward(ob, option)
        return vpred.detach().cpu().numpy()
    def get_tpred(self, ob, option):
        # print(option,type(option))
        ac, vpred, mean,logstd,_,tpred = self.forward(ob, option)
        # print(f"tpred[0].detach().cpu().numpy():{tpred[0].detach().cpu().numpy()}")
        return tpred[0].detach().cpu().numpy()
    def _get_op(self, ob):
        ac, vpred, mean,logstd,op_pi,tpred = self.forward(ob, option = None)
        # print(f"op_pi[0].detach().cpu().numpy():{op_pi[0].detach().cpu().numpy()}")
        return op_pi[0].detach().cpu().numpy()


    def get_option(self, ob):
        # print(f"op_pi:{self.forward(ob, option = None)[4][0]}")
        op_prob = self.forward(ob, option = None)[4].detach().cpu().numpy()[0]
        # print(np.random.choice(len(op_prob), p=op_prob))
        return np.random.choice(len(op_prob), p=op_prob)

    

    def get_opt_adv(self, ob, curr_opt):
        # print(f"ob:{ob}")
        # print(curr_opt)
        vals = [self.forward(ob, torch.tensor([opt]))[1].detach().cpu().numpy() for opt in range(self.num_options)]
        # print(f"vals:{vals}")
        vals = np.stack(vals)
        # print(f"vals:{vals}")
        vals_max = np.amax(vals,axis=0)
        # print(curr_opt)
        return (vals[curr_opt] - vals_max + self.dc), (vals[curr_opt] - vals_max)

   
    
    

     

    