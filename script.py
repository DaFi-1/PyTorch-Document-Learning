import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt



x = torch.ones(3, 3)
y = torch.ones(3, 3, requires_grad=True)
z = x * y
zz = x @ y
print(z.grad_fn)
print(zz.grad_fn)
<MulBackward0 object at 0x7fc422703400>
<MmBackward0 object at 0x7fc422703400>


## Baseados em outros tensores
#zeros_like
#ones_like
#full_like
#as_tensor
#asarray
#
## Sequências
#arange
#range
#linspace
#logspace
#
## Tensores esparsos (Sparse)
#sparse_coo_tensor
#sparse_csr_tensor
#sparse_csc_tensor
#sparse_bsr_tensor
#sparse_bsc_tensor
#
## Matemática especial
#complex
#polar
#heaviside
#
## Avançado (memória/estrutura)
#as_strided
#empty_strided
#
## Quantização
#quantize_per_tensor
#quantize_per_channel
#dequantize
#





































