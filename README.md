
# 📚 Estudo da Documentação do PyTorch

| Métrica                | Valor             |
| ---------------------- | ----------------- |
| Data de Início         | 14 /06 / 2026     |
| Data de Finalizacao    | em progresso      |
| Horas de Estudo        | 3h                |
| Exercícios Realizados  | 0                 |
| Projetos Desenvolvidos | 0                 |
| Tópicos Concluídos     | 0 / 8             |


## Módulo 1 — Fundamentos do PyTorch

* [0. Início Rápido](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) - FlashCard 0 - ✅
* [1. Tensores](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) - FlashCard 1 - ✅
* [2. Conjuntos de Dados e Carregadores de Dados](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* [3. Transforma](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)
* [4. Construir Modelo](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
* [5. Diferenciação Automática](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
* [6. Loop de Otimização](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
* [7. Salvar, Carregar e Usar o Modelo](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)



### 1 - tensores

um tensor e uma estrutura que organizarr nummros em diferentes dimensoes, nao
seria o certo afirma que sao apenas matrizes na pratirca parerce mesmo manipulcao
de matrizes, diferente do NumPy o PyTorch tem aceleracao por GPU ee alta portabilidade
alem de datasets inclusos no ambiente pytorch, o PyTorch tambem ee altametne usado
na area academica & focado nos estudos de deep learning.

para comverte uma array quaquer em um tensor podemos usar o torch.tensor()

```python
# Input
import torch

x = [[1,1],[2,2]]
y = torch.tensor(x)
print(y)

# Output
tensor([[1, 1],
        [2, 2]])
```

podemos tambem converte ndarray do NumPy para Tensores do PyTorch com torch.from_numpy(ndarray)

```python
# Input
import torch
import numpy as np

x = [[1,1],[2,2]]
y = np.array(x)
print(y)
z = torch.from_numpy(y)
print(z)
print('-------------')
print(type(y))
print(type(z))

# Output
[[1 1]
 [2 2]]
tensor([[1, 1],
        [2, 2]])
-------------
<class 'numpy.ndarray'>
<class 'torch.Tensor'>
```
nao podemos esquecer que o tensor & uma classe ee toda classe tem atributos metodos podemos  ver tudo com dir(tensor-OBJETO)

```python
# Input
import torch

x = [[1,1],[2,2]]
z = torch.tensor(x)
print(dir(z))

# Output
['H', 'T', '__abs__', '__add__', '__and__', '__annotations__', '__array__', '__array_priority__', '__arr
ay_wrap__', '__bool__', '__class__', '__complex__', '__contains__', '__deepcopy__', '__delattr__', '__de
litem__', '__dict__', '__dir__', '__div__', '__dlpack__', '__dlpack_c_exchange_api__', '__dlpack_device_
_', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__get
item__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__i
lshift__', '__imod__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__'
, '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '_
_len__', '__long__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__',
 '__neg__', '__new__', '__nonzero__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdiv__'
, '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul_
_', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv_
_', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__sub
classhook__', '__torch_dispatch__', '__torch_function__', '__truediv__', '__weakref__', '__xor__', '_add
mm_activation', '_autocast_to_full_precision', '_autocast_to_reduced_precision', '_backward_hooks', '_ba
se', '_cdata', '_clear_non_serializable_cached_data', '_coalesced_', '_conj', '_conj_physical', '_dimI',
 '_dimV', '_dtensor__new__', '_fix_weakref', '_grad', '_grad_fn', '_has_symbolic_sizes_strides', '_indic
es', '_is_all_true', '_is_any_true', '_is_view', '_is_zerotensor', '_lazy_clone', '_make_subclass', '_ma
ke_wrapper_subclass', '_neg_view', '_nested_tensor_size', '_nested_tensor_storage_offsets', '_nested_ten
sor_strides', '_nnz', '_philox_normal_', '_philox_uniform_', '_post_accumulate_grad_hooks', '_python_dis
patch', '_reduce_ex_internal', '_rev_view_func_unsafe', '_sparse_mask_projection', '_to_dense', '_to_spa
rse', '_to_sparse_bsc', '_to_sparse_bsr', '_to_sparse_csc', '_to_sparse_csr', '_typed_storage', '_update
_names', '_use_count', '_values', '_version', '_view_func', '_view_func_unsafe', 'abs', 'abs_', 'absolut
e', 'absolute_', 'acos', 'acos_', 'acosh', 'acosh_', 'add', 'add_', 'addbmm', 'addbmm_', 'addcdiv', 'add
cdiv_', 'addcmul', 'addcmul_', 'addmm', 'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 'adjoint', 'align_
as', 'align_to', 'all', 'allclose', 'amax', 'amin', 'aminmax', 'angle', 'any', 'apply_', 'arccos', 'arcc
os_', 'arccosh', 'arccosh_', 'arcsin', 'arcsin_', 'arcsinh', 'arcsinh_', 'arctan', 'arctan2', 'arctan2_'
, 'arctan_', 'arctanh', 'arctanh_', 'argmax', 'argmin', 'argsort', 'argwhere', 'as_strided', 'as_strided
_', 'as_strided_scatter', 'as_subclass', 'asin', 'asin_', 'asinh', 'asinh_', 'atan', 'atan2', 'atan2_', 
'atan_', 'atanh', 'atanh_', 'backward', 'baddbmm', 'baddbmm_', 'bernoulli', 'bernoulli_', 'bfloat16', 'b
incount', 'bitwise_and', 'bitwise_and_', 'bitwise_left_shift', 'bitwise_left_shift_', 'bitwise_not', 'bi
twise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_right_shift', 'bitwise_right_shift_', 'bitwise_xor', 
'bitwise_xor_', 'bmm', 'bool', 'broadcast_to', 'byte', 'cauchy_', 'ccol_indices', 'cdouble', 'ceil', 'ce
il_', 'cfloat', 'chalf', 'char', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk', 'clamp', 'cl
amp_', 'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'clip', 'clip_', 'clone', 'coalesce', 'col_
indices', 'conj', 'conj_physical', 'conj_physical_', 'contiguous', 'copy_', 'copysign', 'copysign_', 'co
rrcoef', 'cos', 'cos_', 'cosh', 'cosh_', 'count_nonzero', 'cov', 'cpu', 'cross', 'crow_indices', 'cuda',
 'cummax', 'cummin', 'cumprod', 'cumprod_', 'cumsum', 'cumsum_', 'data', 'data_ptr', 'deg2rad', 'deg2rad
_', 'dense_dim', 'dequantize', 'det', 'detach', 'detach_', 'device', 'diag', 'diag_embed', 'diagflat', '
diagonal', 'diagonal_scatter', 'diff', 'digamma', 'digamma_', 'dim', 'dim_order', 'dist', 'div', 'div_',
 'divide', 'divide_', 'dot', 'double', 'dsplit', 'dtype', 'eig', 'element_size', 'eq', 'eq_', 'equal', '
erf', 'erf_', 'erfc', 'erfc_', 'erfinv', 'erfinv_', 'exp', 'exp2', 'exp2_', 'exp_', 'expand', 'expand_as
', 'expm1', 'expm1_', 'exponential_', 'fill_', 'fill_diagonal_', 'fix', 'fix_', 'flatten', 'flip', 'flip
lr', 'flipud', 'float', 'float_power', 'float_power_', 'floor', 'floor_', 'floor_divide', 'floor_divide_
', 'fmax', 'fmin', 'fmod', 'fmod_', 'frac', 'frac_', 'frexp', 'gather', 'gcd', 'gcd_', 'ge', 'ge_', 'geo
metric_', 'geqrf', 'ger', 'get_device', 'grad', 'grad_dtype', 'grad_fn', 'greater', 'greater_', 'greater
_equal', 'greater_equal_', 'gt', 'gt_', 'half', 'hardshrink', 'has_names', 'hash_tensor', 'heaviside', '
heaviside_', 'histc', 'histogram', 'hsplit', 'hypot', 'hypot_', 'i0', 'i0_', 'igamma', 'igamma_', 'igamm
ac', 'igammac_', 'imag', 'index', 'index_add', 'index_add_', 'index_copy', 'index_copy_', 'index_fill', 
'index_fill_', 'index_put', 'index_put_', 'index_reduce', 'index_reduce_', 'index_select', 'indices', 'i
nner', 'int', 'int_repr', 'inverse', 'ipu', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'i
s_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia'
, 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_qua
ntized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan
', 'is_xla', 'is_xpu', 'isclose', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal', 'istft
', 'item', 'itemsize', 'kron', 'kthvalue', 'layout', 'lcm', 'lcm_', 'ldexp', 'ldexp_', 'le', 'le_', 'ler
p', 'lerp_', 'less', 'less_', 'less_equal', 'less_equal_', 'lgamma', 'lgamma_', 'log', 'log10', 'log10_'
, 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'log_normal_', 'log_softmax', 'logaddexp', 'logaddexp2', '
logcumsumexp', 'logdet', 'logical_and', 'logical_and_', 'logical_not', 'logical_not_', 'logical_or', 'lo
gical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logsumexp', 'long', 'lstsq', 'lt', 'lt_',
 'lu', 'lu_solve', 'mH', 'mT', 'map2_', 'map_', 'masked_fill', 'masked_fill_', 'masked_scatter', 'masked
_scatter_', 'masked_select', 'matmul', 'matrix_exp', 'matrix_power', 'max', 'maximum', 'mean', 'median',
 'min', 'minimum', 'mm', 'mode', 'module_load', 'moveaxis', 'movedim', 'msort', 'mtia', 'mul', 'mul_', '
multinomial', 'multiply', 'multiply_', 'mv', 'mvlgamma', 'mvlgamma_', 'name', 'names', 'nan_to_num', 'na
n_to_num_', 'nanmean', 'nanmedian', 'nanquantile', 'nansum', 'narrow', 'narrow_copy', 'nbytes', 'ndim', 
'ndimension', 'ne', 'ne_', 'neg', 'neg_', 'negative', 'negative_', 'nelement', 'new', 'new_empty', 'new_
empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'nextafter', 'nextafter_', 'nonzero',
 'nonzero_static', 'norm', 'normal_', 'not_equal', 'not_equal_', 'numel', 'numpy', 'orgqr', 'ormqr', 'ou
ter', 'output_nr', 'permute', 'pin_memory', 'pinverse', 'polygamma', 'polygamma_', 'positive', 'pow', 'p
ow_', 'prelu', 'prod', 'put', 'put_', 'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_
points', 'q_scale', 'q_zero_point', 'qr', 'qscheme', 'quantile', 'rad2deg', 'rad2deg_', 'random_', 'rave
l', 'real', 'reciprocal', 'reciprocal_', 'record_stream', 'refine_names', 'register_hook', 'register_pos
t_accumulate_grad_hook', 'reinforce', 'relu', 'relu_', 'remainder', 'remainder_', 'rename', 'rename_', '
renorm', 'renorm_', 'repeat', 'repeat_interleave', 'requires_grad', 'requires_grad_', 'reshape', 'reshap
e_as', 'resize', 'resize_', 'resize_as', 'resize_as_', 'resize_as_sparse_', 'resolve_conj', 'resolve_neg
', 'retain_grad', 'retains_grad', 'roll', 'rot90', 'round', 'round_', 'row_indices', 'rsqrt', 'rsqrt_', 
'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'scatter_reduce', 'scatter_reduce_', 'select', 'se
lect_scatter', 'set_', 'sgn', 'sgn_', 'shape', 'share_memory_', 'short', 'sigmoid', 'sigmoid_', 'sign', 
'sign_', 'signbit', 'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'size', 'slice_inverse', 'slice_sca
tter', 'slogdet', 'smm', 'softmax', 'solve', 'sort', 'sparse_dim', 'sparse_mask', 'sparse_resize_', 'spa
rse_resize_and_clear_', 'split', 'split_with_sizes', 'sqrt', 'sqrt_', 'square', 'square_', 'squeeze', 's
queeze_', 'sspaddmm', 'std', 'stft', 'storage', 'storage_offset', 'storage_type', 'stride', 'sub', 'sub_
', 'subtract', 'subtract_', 'sum', 'sum_to_size', 'svd', 'swapaxes', 'swapaxes_', 'swapdims', 'swapdims_
', 'symeig', 't', 't_', 'take', 'take_along_dim', 'tan', 'tan_', 'tanh', 'tanh_', 'tensor_split', 'tile'
, 'to', 'to_dense', 'to_mkldnn', 'to_padded_tensor', 'to_sparse', 'to_sparse_bsc', 'to_sparse_bsr', 'to_
sparse_coo', 'to_sparse_csc', 'to_sparse_csr', 'tolist', 'topk', 'trace', 'transpose', 'transpose_', 'tr
iangular_solve', 'tril', 'tril_', 'triu', 'triu_', 'true_divide', 'true_divide_', 'trunc', 'trunc_', 'ty
pe', 'type_as', 'unbind', 'unflatten', 'unfold', 'uniform_', 'unique', 'unique_consecutive', 'unsafe_chu
nk', 'unsafe_split', 'unsafe_split_with_sizes', 'unsqueeze', 'unsqueeze_', 'untyped_storage', 'values', 
'var', 'vdot', 'view', 'view_as', 'vsplit', 'where', 'xlogy', 'xlogy_', 'xpu', 'zero_']
```

podemos explor 3 atributos que mostra nessa parte da documentacao .shape,.dtype,.device
```python
# Input
#.shape  - dimensões
#.dtype  - tipo do tensor
#.device - mostra onde foi processado
import torch

x = [[1,1],[2,2]]
z = torch.tensor(x)
print(z.shape) j
print(z.dtype)
print(z.device)

# Output
torch.Size([2, 2])
torch.int64
cpu
```

podemos tambem concatenar tensores com torch.cat(), torch.stack()

```python
# Input
import torch

x = torch.ones(2,2)
y = torch.full((2,2),3)
print(x, '\n', y) 
print(torch.cat([x,y],dim=1)) # junta tensores na mesma dimensao
print(torch.stack([x,y])) # empilha o ternsor em uma nova dimensao 

# Output
tensor([[1., 1.],
        [1., 1.]]) 
tensor([[3, 3],
        [3, 3]])
tensor([[1., 1., 3., 3.],
        [1., 1., 3., 3.]])
tensor([[[1., 1.],
         [1., 1.]],

        [[3., 3.],
         [3., 3.]]])
```

mutiplicacao de tensorres com * e @, vamos fazer um codig osimpels que multiplica
dois tensores com * e @ , algo bem simplorio, o reto eee alto explicativo para qeum
estudou algebra linear

```python
# Input
import torch

x = torch.ones(2,2)
y = torch.full((2,2),3.0)
print(x * y)
print(x @ y)

# Output
tensor([[3., 3.],
        [3., 3.]])
tensor([[6., 6.],
        [6., 6.]])
```

in-place operation, sao operacoes seguidas de "_" ee elas guardan ses resutados no operando

```python
# Input
import torch

x = torch.ones(2,2)
print(f"{x} \n")
x.add_(5)
print(x)

# Output
tensor([[1., 1.],
        [1., 1.]]) 

tensor([[6., 6.],
        [6., 6.]])
```
indexacao eee fatiamento

```python
# Input
# Output
Resumo mental rápido
x[i] → elemento
x[i:j] → fatia
x[:, j] → coluna
x[i, :] → linha
x[mask] → filtro
```

### 🧪 Exercícios de Tensors (PyTorch)

> ⚠️ DISCLAIMER / AVISO IMPORTANTE  
>  
> Os exercícios e o mini projeto apresentados neste material foram gerados por Inteligência Artificial (IA), com base no conteúdo estudado sobre o tópico de **Tensors no PyTorch**, conforme descrito na documentação oficial.  
>  
> O objetivo deste material é exclusivamente educacional, servindo como prática complementar para reforço do aprendizado dos conceitos abordados na documentação.  
>  
> ✔️ Conteúdo baseado em:  
> - Operações com tensores  
> - Inicialização e tipos de dados  
> - Indexação e slicing  
> - Operações matemáticas  
> - Redução de dimensões  
> - Integração com NumPy  
> - Manipulação de memória  
>  
> 📌 Este material não substitui a documentação oficial, mas foi estruturado para facilitar o estudo prático e progressivo do tema.

---

#### 📌 Exercício 1: Inicialização e Atributos

Crie uma lista Python bidimensional \(3 \times 3\) com valores inteiros sequenciais (de 1 a 9).

Converta essa lista diretamente em um tensor PyTorch.

Imprima as três propriedades fundamentais desse tensor:
- shape (formato)
- dtype (tipo de dado)
- device (dispositivo)

---

#### 📌 Exercício 2: Tipos e Clonagem de Estruturas

Usando o tensor do Exercício 1:

1. Crie um tensor de zeros com o mesmo shape e dtype  
2. Crie um tensor aleatório com o mesmo shape, mas force dtype float32  

---

#### 📌 Exercício 3: Indexação e Slicing Avançado

Crie um tensor 5x5 preenchido com 2.0.

Depois:
- troque a 3ª coluna por 0.0  
- troque a última linha por 9.0  

---

#### 📌 Exercício 4: Operações Aritméticas e Memória

Crie dois tensores 3x3 aleatórios.

Faça multiplicação matricial de 3 formas:
- operador `@`
- `.matmul()`
- usando `out=`

Depois faça multiplicação elemento a elemento.

---

#### 📌 Exercício 5: Redução e In-place

Crie um tensor 4x4 com 1.0.

Depois:
- some 4.0 usando operação in-place  
- calcule a soma total  
- converta o resultado para um float Python  

---

#### 📌 Exercício 6: Ponte PyTorch ↔ NumPy

##### Parte 1: Tensor → NumPy

Crie um tensor de 5 elementos com 0.0.

Converta para NumPy e verifique o compartilhamento de memória.

Modifique o tensor em-place e observe o efeito no NumPy.

---

##### Parte 2: NumPy → Tensor

Crie um array NumPy.

Converta para tensor PyTorch.

Modifique o NumPy usando `np.add(..., out=...)` e observe o impacto no tensor.
























































```python
# Input
# Output
```


```python
# Input
# Output
```












# Flash cards  - PyTorch

## FlashCard - 1

```python

o que um tensor -------------------------------------------------------- uma estrutura que organiza números em diferentes dimensões
o que e CUDA ? --------------------------------------------------------- plataforma da NVIDIA para usar GPU em computação
o que e ROCm ? --------------------------------------------------------- plataforma da AMD equivalente ao CUDA, usada em GPUs AMD
Diferenca entre NUMPY & PYTORCH ? -------------------------------------- NumPy é CPU only e sem autograd; PyTorch tem CPU+GPU e autograd
sintaxe de fatiamento em tensores -------------------------------------- tensor[início:fim:passo]
operações in-places são indicadas por um ------------------------------- "_"
Todo Tensorr & umm objeto ---------------------------------------------- sim
Um tensor no pytorch e um objeto --------------------------------------- sim
Operações in-place armazenam o resultado no ---------------------------- operando
Como criar um tensor atravesi de arrays -------------------------------- torch.tensor()
Como criar um tensor atravesi de ndarray ------------------------------- torch.from_numpy(ndarray)
Tensor com numero desejado --------------------------------------------- torch.full()
Duas formas de multiplicar tensores (element-wise) --------------------- t1 * t2, t1.mul(t2)
Duas formas de multiplicar tensores (matrix-multiplication) ------------ t1 @ t2, t1.matmul(t2)

Como concatenar tensores ----------------------------------------------- torch.cat()   - junta tensores na mesma dimensão
                                                                         torch.stack() - empilha em uma nova dimensão

3 atributos de um tensor ----------------------------------------------- .shape  - dimensões
                                                                         .dtype  - tipo de dado
                                                                         .device - onde foi executado (CPU ou GPU)

```



















