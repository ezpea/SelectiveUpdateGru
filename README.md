# SelectiveUpdateGru

A custom GRU (Gated Recurrent Unit) that selectively updates its hidden 
state based on a control signal.

## Installation

```bash
pip install git+https://github.com/yourusername/SelectiveUpdateGru.git
```

## Usage

```python
import torch
from selective_update_gru import SelectiveUpdateGru

input_size = 5
hidden_size = 3
num_layers = 1
model = SelectiveUpdateGru(input_size, hidden_size, num_layers)

x = torch.randn(2, 4, input_size)
out, h = model(x)
print(out, h)
```
