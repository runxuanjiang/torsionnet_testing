import torch
import torch_geometric.nn as gnn

set2set = gnn.Set2Set(128, processing_steps=2)
for i in range(1000):
    input = torch.randn(26, 128)
    output = set2set(input)
