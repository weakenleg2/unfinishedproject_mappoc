
import numpy as np

import torch
import torch.nn as nn
#两大类，第一类去量纲,例如（x-xmin）/（xmax-x），第二类就是范数

class ValueNorm(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        # a learnable parameter of the model, but it will not be updated via gradient
        #  descent during the training process.
        # This represents the accumulated mean of the observations. 
        # It's updated incrementally, meaning that with every new batch of data, 
        # the running mean is adjusted to include the new information. This adjustment 
        # is controlled by a decay factor (beta), which determines the weightage given 
        # to the old accumulated mean versus the new batch's mean.
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # 去偏差项
        
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()
        # Resets running statistics to zero.

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        # debiasing_term
        # 该术语是一个累加器，用于跟踪对当前运行平均值有贡献的有效观测值（或权重）数量。 
        # 最初，它从零开始，并随着每次更新而递增。 增量的大小按因子 (1 - beta) 缩小，
        # 就像运行平均值的更新一样。 去偏项可以平衡运行平均值的指数衰减特性。
        # 除法是为了去偏差，clamp是为了保证debiasing这个值不会太小
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        # 该行使用去偏均值和去偏均方值计算观测值的去偏方差
        # var(x)=E(x^2)-E(x)^2
        return debiased_mean, debiased_var
        # Returns the debiased mean and variance of the running statistics.

    @torch.no_grad()
    # This ensures that the operations within the method do not track gradients.
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)  
        # not elegant, but works in most cases
        # The next line moves the input_vector tensor to the same device as self.running_mean.
        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
            # 如果 per_element_update 为 True，则根据批量大小调整权重。
            #  这在批量大小变化且您希望保持一致的衰减率的情况下非常有用。
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        # the running statistics are being updated using an exponential moving average
        # beta*current_val+(1-beta)*past
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        # For instance, if input_vector has a shape of (batch_size, num_features) 
        # and self.norm_axes is 1, then mean and torch.sqrt(var) might initially 
        # have a shape of (num_features,). The indexing operation will change their 
        # shapes to (1, num_features), allowing them to be broadcasted correctly over
        #  the batch_size dimension when performing the arithmetic operations with 
        # input_vector.
        
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()
        
        return out
