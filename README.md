# SelectiveUpdateGru

A custom GRU (Gated Recurrent Unit) that selectively updates its hidden 
state based on a control signal.

## Installation

```bash
pip install git+https://github.com/ezpea/SelectiveUpdateGru.git
```

## Usage

```python
import torch
from selective_update_gru import SelectiveUpdateGru
torch.manual_seed(20240907)

input_size = 5
hidden_size = 3
num_layers = 1
model = SelectiveUpdateGru(input_size, hidden_size, num_layers)

# Example data loader (replace with your actual data)
num_sequences = 10
seq_length = 10
batch_size = 2

x = torch.cat((
        torch.randn(batch_size, seq_length, 1) > 0, # Control signal
        torch.randn(batch_size, seq_length, input_size-1)), 
        dim=2)

out, h = model(x)
print(out, h)
```

```txt
tensor([[[ 0.0000, -0.2103, -0.0151],
         [ 0.0000, -0.1589,  0.1279],
         [ 1.0000, -0.1589,  0.1279],
         [ 0.0000, -0.2203,  0.4342],
         [ 0.0000, -0.1311,  0.1062],
         [ 1.0000, -0.1311,  0.1062],
         [ 1.0000, -0.1311,  0.1062],
         [ 1.0000, -0.1311,  0.1062],
         [ 1.0000, -0.1311,  0.1062],
         [ 0.0000, -0.0087,  0.1640]],

        [[ 1.0000,  0.0000,  0.0000],
         [ 0.0000, -0.3677,  0.5259],
         [ 1.0000, -0.3677,  0.5259],
         [ 0.0000, -0.4960,  0.5107],
         [ 0.0000, -0.3328,  0.4307],
         [ 1.0000, -0.3328,  0.4307],
         [ 0.0000, -0.1299, -0.0719],
         [ 0.0000, -0.1653,  0.4407],
         [ 1.0000, -0.1653,  0.4407],
         [ 0.0000,  0.2851, -0.0926]]], grad_fn=<TransposeBackward1>) tensor([[[ 0.0000, -0.0087,  0.1640],
         [ 0.0000,  0.2851, -0.0926]]], grad_fn=<StackBackward0>)

```