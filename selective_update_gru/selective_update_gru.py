"""
Implements SelectiveUpdateGru class
"""

from torch import nn


class SelectiveUpdateGru(nn.Module):
    """
    SelectiveUpdateGru - A custom GRU (Gated Recurrent Unit) that selectively
    updates its hidden state based on a control signal.

    This GRU takes in batches of sequences where each sequence has multiple
    features. The first feature is a control signal that must be either 0 or 1:
    - When the control signal is 1, the GRU ignores the inputs for that time step
    and preserves the previous state and output. When the control signal is 0,
    the GRU processes the remaining input features as usual.
    - The control signal is propagated as the first feature of the hidden
    state.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers.

    Attributes:
        gru (nn.GRU): The GRU layer.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        my_inf = 1e22

        for n, p in self.named_parameters():
            # print(n, p.shape)
            # Custom initialization for selective update behavior
            if n.startswith("gru.weight_ih_l"):
                # Initialize input-hidden weights
                p.data[:, 0] = 0

                # W_iz - first input is (0,1) if it is 1 will make z(1:)=1 and z(0)=0
                # so h(1:,t) = h(1:,t-1) and h(0,t) = the input
                p.data[hidden_size + 1 : 2 * hidden_size, 0] = (
                    my_inf  # W_iz[:,1:] remember h[t-1,1:] if input is 1
                )
                p.data[hidden_size, 0] = (
                    0  # W_iz[:,1:] don't remember h[t-1,0] if input is 1 or 0 (see bias)
                )

                # W_in - make n[t,0] = x[t,0]
                p.data[2 * hidden_size, 0] = my_inf  # W_in[0,0]
                p.data[2 * hidden_size, 1:] = 0

                def _zero_grad_hook(grad):
                    grad[..., 0] = 0
                    grad[..., 2 * hidden_size, :] = 0
                    return grad

                p.register_hook(_zero_grad_hook)

            elif n.startswith("gru.bias_ih_l"):
                # Initialize input-hidden biases
                p.data[0] = -1e12  # b_ir[0]=-inf so r[...,0]=0
                p.data[hidden_size] = -my_inf  # b_iz[0]=-inf so z[...,0] = 0
                p.data[2 * hidden_size] = (
                    0  # b_in[0] = 0 given W_in[0,0]=inf and W_in[0,1:]=0
                )

                def _zero_grad_hook(grad):
                    grad[..., 0] = 0
                    grad[..., hidden_size] = 0
                    grad[..., 2 * hidden_size] = 0
                    return grad

                p.register_hook(_zero_grad_hook)

    def forward(self, x):
        """
        Forward pass for the SelectiveUpdateGru.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, sequence_length, hidden_size).
            h (torch.Tensor): Hidden state tensor of shape (num_layers, batch_size, hidden_size).
        """
        out, h = self.gru(x)
        return out, h
