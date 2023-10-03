import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import init, get_clones
from torch.distributions import Categorical, Normal


# not pretty sure do we really need to track the name
def dense3D2_torch(x, size, name, option, num_options=1, weight_init=None, bias=True):
    if weight_init is None:
        weight_init = torch.nn.init.orthogonal_
    weights = []
    biases = []
    w = torch.empty(num_options, x.shape[1], size)
    # [number of options, input size, number of neurons]
    weight_init(w)
    weights[name + "/w"] = w
    
    # Select the weight matrix for the current option and perform matrix multiplication
    ret = torch.mm(x, w[option[0]])
    # x=[input size, number of neurons]
    # If bias is enabled, define the bias tensor and add it to the result
    if bias:
        b = torch.zeros(num_options, size)
        ret += b[option[0]]
        biases[name + "/b"] = b
        # Each option has its own 1D bias vector of size [number of neurons]
    
    return ret

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

class MlpBase(nn.Module):
    def __init__(self, args, q_space_dim, ac_space_dim, pi_space_dim, mu_space_dim, num,obs_shape,hidden_size):

        super(MlpBase, self).__init__()
        gaussian_fixed_var=True
        recurrent = False
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        num_options=2
        dc=0, 
        hid_size =  args.layer_N
        num_hid_layers= hidden_size
        self.num=int(num)

        self.ac_space_dim = ac_space_dim
        self.q_space_dim = q_space_dim
        self.pi_space_dim = pi_space_dim
        self.mu_space_dim = mu_space_dim
        self.num_options = num_options
        self.dc = dc
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_shape[0])
        self.ob_rms_q = RunningMeanStd(shape=(q_space_dim,))
        self.ob_rms_pi = RunningMeanStd(shape=(pi_space_dim,))
        self.ob_rms_mu = RunningMeanStd(shape=(mu_space_dim,))

        self.q_net0 = MLPLayer(q_space_dim, hid_size, num_hid_layers, 1)
        self.q_net1 = MLPLayer(q_space_dim, hid_size, num_hid_layers, 1)

        self.option_net0 = MLPLayer(mu_space_dim, hid_size, num_hid_layers, 1)
        self.option_net1 = MLPLayer(mu_space_dim, hid_size, num_hid_layers,1)

        self.pi_net = MLPLayer(pi_space_dim, hid_size, num_hid_layers, ac_space_dim)
        self.logstd = nn.Parameter(torch.zeros(ac_space_dim))

    def forward(self, ob, option, stochastic=True):
        obz_q = torch.clamp((ob[:, 3:self.q_space_dim + 3] - self.ob_rms_q.mean) / self.ob_rms_q.std, -10.0, 10.0)
        obz_pi = torch.clamp((ob[:, 3:self.pi_space_dim + 3] - self.ob_rms_pi.mean) / self.ob_rms_pi.std, -10.0, 10.0)
        obz_mu = torch.clamp((ob[:, :self.mu_space_dim] - self.ob_rms_mu.mean) / self.ob_rms_mu.std, -10.0, 10.0)

        q_val0 = self.q_net0(obz_q)
        q_val1 = self.q_net1(obz_q)

        vpred = q_val1 if option[0] == 1 else q_val0

        option_val0 = self.option_net0(obz_mu)
        option_val1 = self.option_net1(obz_mu)
        option_vals = torch.cat([option_val0, option_val1], dim=1)
        op_pi = F.softmax(option_vals, dim=1)

        mean = self.pi_net(obz_pi)
        std = torch.exp(self.logstd)
        dist = Normal(mean, std)
        ac = dist.sample() if stochastic else mean
        ac = torch.clamp(ac, -1.0, 1.0)

        return ac, vpred, option_vals, std

    def get_option(self, ob):
        obz_mu = torch.clamp((ob[:, :self.mu_space_dim] - self.ob_rms_mu.mean) / self.ob_rms_mu.std, -10.0, 10.0)
        option_val0 = self.option_net0(obz_mu)
        option_val1 = self.option_net1(obz_mu)
        option_vals = torch.cat([option_val0, option_val1], dim=1)
        op_pi = F.softmax(option_vals, dim=1)
        return torch.multinomial(op_pi, 1).squeeze().item()