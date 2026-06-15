
import torch

x = torch.ones(2,2)
print(f"{x} \n")
x.add_(5)
print(x)

#########################
#Indexação e fatiamento padrão de tensores
#
#x_data = torch.ones(5,5)
#x_data[:,0] =0
#print(x_data)

#########################
# Objeto Tensor
# 
# .shape
# .dtype
# .device
#

#########################
# Tensor Creation Ops
## Criação básica
#tensor
#zeros
#ones
#full
#empty
#eye
#
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
## Conversão de dados externos
#from_numpy
#from_dlpack
#frombuffer
#from_file
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





































