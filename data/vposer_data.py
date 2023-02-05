import sys
import numpy as np
import torch
torch.manual_seed(0)

sys.path.append('/Users/sele/models/vposer/')
from vposer_smpl import VPoser

vposer_weights_path = '/Users/sele/models/vposer/snapshots/TR00_E096.pt'
vposer = VPoser(num_neurons=512, latentD=32, data_shape=(1, 21, 3), use_cont_repr=True)
vposer.load_state_dict(torch.load(vposer_weights_path, map_location=torch.device('cpu')))
vposer.eval()

weight_dict = dict()
for name, weights in vposer.named_parameters():
    weight_dict[name] = weights.detach().numpy().astype(np.float64)
for name, weights in vposer.named_buffers():
    weight_dict[name] = weights.detach().numpy().astype(np.float64)
np.savez('vposer_weights.npz', **weight_dict)

# Example for decoding for testing.
z = torch.randn(32)
R_out = vposer.decode(z.clone()).view(21, 3, 3)
np.savez(
    'test/vposer_data.npz',
    z=z.detach().numpy().astype(np.float64),
    R_out=torch.flatten(R_out, start_dim=-2).detach().numpy().astype(np.float64),
)

# Example for linear layer testing.
l = torch.nn.Linear(3, 2)
z = torch.rand(3)
out = l(z)
weight_dict = {
    "layer.weight": l.weight.detach().numpy().astype(np.float64), 
    "layer.bias": l.bias.detach().numpy().astype(np.float64)
}
np.savez(
    'test/linear_layer.npz',
    z=z.detach().numpy().astype(np.float64),
    out=out.detach().numpy().astype(np.float64),
    **weight_dict,
)

# Example for leaky relu testing.
z = torch.rand(20) - 0.5
out = torch.nn.functional.leaky_relu(z, negative_slope=0.2)
np.savez(
    'test/leaky_relu.npz',
    z=z.detach().numpy().astype(np.float64),
    out=out.detach().numpy().astype(np.float64),
)

# Example for 1D batch norm testing.
z = torch.rand(20) + 0.5
l = torch.nn.BatchNorm1d(20).eval()
out = l(z[None, :])
weight_dict = {
    "layer.weight": l.weight.detach().numpy().astype(np.float64), 
    "layer.bias": l.bias.detach().numpy().astype(np.float64),
    "layer.running_mean": l.running_mean.detach().numpy().astype(np.float64),
    "layer.running_var": l.running_var.detach().numpy().astype(np.float64),
}
np.savez(
    'test/batch_norm.npz',
    z=z.detach().numpy().astype(np.float64),
    out=out.detach().numpy().astype(np.float64),
    **weight_dict,
)
