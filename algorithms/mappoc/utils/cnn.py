import torch.nn as nn
from .util import init

"""CNN Modules and utils."""
# 卷积操作的本质是对而未知图案的局部和标准X图案的局部一
# 个一个比对时的计算过程，便是卷积操作。卷积计算结果为1表示匹配，否则不匹配。
# 对图像（不同的数据窗口数据）和滤波矩阵（一组固定的权重：因为每个神经元的多个权重固定，
# 所以又可以看做一个恒定的滤波器filter）做内积（逐个元素相乘再求和）的操作就是所谓的『卷积』操作
# a.深度depth：神经元个数，决定输出的depth厚度。同时代表滤波器个数。
# b. 步长stride：决定滑动多少步可以到边缘。
# c. 填充值zero-padding：在外围边缘补充若干圈0，方便从初始位置以步长为单位可以刚好滑倒末尾位置，
# 通俗地讲就是为了总长能被步长整除。
# 池化，简言之，即取区域平均或最大

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    # Suppose x has a shape of [batch_size, channels, height, width],
    #  such as [32, 3, 64, 64] (i.e., a batch of 32 images, each with 3 
    # channels and 64x64 resolution). then, after that it is 32*(3*64*64),2 dim


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])
        # The gain is a scaling factor that can be used with certain weight 
        # initialization methods to ensure that weights are initialized with
        #  the right scale

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        # This function call is using the init function (which seems to be imported or 
        # defined elsewhere in your code) to initialize the module m.
        # init_method: Specifies the method to initialize the weights of the module.
        #  This could be something like nn.init.xavier_uniform_ or nn.init.orthogonal_,
        #  based on the use_orthogonal flag from the outer function/context.
        # lambda x: nn.init.constant_(x, 0): This is an anonymous function (lambda function)
        #  that initializes the biases of the module to zeros using nn.init.constant_.
        # gain=gain: Specifies the gain value for the weight initialization. This is 
        # based on the activation function used in the network and is calculated using
        #  nn.init.calculate_gain.

        # nn.init.orthogonal_ is a method in PyTorch used for weight initialization.
        #  It initializes the weight tensor of a layer with an orthogonal matrix.\
        #  在神经网络的背景下，这有助于保留输入数据流经各层时的方差
        # 有利于避免梯度消失和爆炸，有利于RNN网络

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            # The number of input features to this layer is calculated based on the output
            #  size of the convolutional layer. It considers factors like input width, 
            # input height, kernel size, and stride. 两个线性层是全连接层
            init_(nn.Linear(hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            hidden_size)
                  ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), 
            active_func)
            # Start with a 2D convolution layer to extract spatial features from the input.
            # Apply an activation function.
            # Flatten the 2D spatial data to 1D.
            # Pass the flattened data through two fully connected (linear) 
            # layers with activation functions in between.

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x
