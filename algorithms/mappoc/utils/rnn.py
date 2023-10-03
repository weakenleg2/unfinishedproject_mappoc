import torch
import torch.nn as nn

"""RNN modules."""
# 关键区别就是其会学习自身不同时刻的输出
# 1. 这里的W,U,V在每个时刻都是相等的(权重共享).
#  2. 隐藏状态可以理解为:  S=f(现有的输入+过去记忆总结)

# implementation of the Gated Recurrent Unit (GRU). 
# The GRU is a type of recurrent neural network (RNN) architecture.
# The method also handles sequences of varying lengths by using masks (masks). 
# Masks are binary tensors that indicate whether a particular sequence is valid 
# at a specific timestep. This is particularly useful for batches of sequences 
# with different lengths, a common situation when working with sequences in deep learning
class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
            # batch, num_layers, hidden_size)

            # masks.repeat(1, self._recurrent_N): The mask is being repeated across the recurrent dimension
            # . This ensures that the mask has the same size as the hidden states.
            # .unsqueeze(-1): Adds an extra dimension to the end of the tensor. This is to ensure that
            #  the mask can be broadcasted to the hidden state tensor for element-wise multiplication.
            # hxs * ...: Element-wise multiplication of the hidden states with the mask. This operation 
            # effectively resets the hidden state for sequences that have ended.
            # .transpose(0, 1): This transposes the first and second dimensions. RNNs in PyTorch expect
            #  the hidden state in the shape (num_layers, batch, hidden_size).
            # .contiguous(): Ensures that the tensor is stored in a contiguous block of memory, necessary
            # for transpose 
            # RNNs in PyTorch expect input in the shape (seq_len, batch, input_size),
            #  where:

            # seq_len is the length of the sequence.
            # batch is the batch size.
            # input_size is the number of features at each timestep.
            # By using unsqueeze(0), we're preparing the tensor to have a sequence length of 1.
        # If the first dimension of x (usually batch size) is equal to the first dimension of 
        # the hidden state hxs, then:

        # The input x is reshaped and passed through the GRU along with the hidden state.
        # The output and hidden state are processed accordingly.
        # If not:

        # The input x is reshaped, and sequences that don't have any zeros in their
        #  masks are grouped together.
        # These sequences are processed in the GRU in batches, which can be more efficient.
        # The outputs are then concatenated together to form the final output.
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            # batch
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())
            # masks[1:]: Skips the first timestep since we assume that the first timestep 
            # always has valid data for all sequences.
            # == 0.0: Checks where the mask is zero, indicating the end of a sequence.
            # 检查至少有没有一个是0,如果有,一个any会直接返回TRUE.总之会使得返回值更精炼
            # .any(dim=-1): Checks if any element in the last dimension is zero.
            # .nonzero(): Gets the indices of non-zero elements.
            # .squeeze(): Removes any singleton dimensions.
            # .cpu(): Moves the tensor to the CPU.
            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            # This block adjusts the indices since we skipped the first timestep earlier.
            # If has_zeros is   a scalar, it's converted to a list. Otherwise, 
            # it's incremented by 1 and converted to a list.

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs
