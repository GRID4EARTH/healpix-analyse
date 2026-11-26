# Recipe: GAN critic with a gradient penalty (double backward)

A WGAN-GP: the critic loss includes `(‖∂critic/∂x‖ − 1)²` at points between
real and generated fields. The penalty takes the gradient of the critic with
`torch.autograd.grad(..., create_graph=True)`, and the critic loss is then
differentiated again with respect to the weights: a second-order gradient
through every layer. `HealPixConv` (stride 1 and 2) runs it on its own
kernels, Triton on CUDA, with the same determinism as the first backward;
`torch.func` (`vmap`, `jacrev`, `jacfwd`, `jvp`) works on the layer too.

Points to copy:

- `create_graph=True` in the inner `torch.autograd.grad`, then a plain
  `loss.backward()`;
- the critic goes one level coarser with `HealPixConv(stride=2)`, and the
  next layer is built on `critic.c2.cell_ids_out`;
- `global_pool` reduces the cells to one vector per sample;
- the same pattern gives any loss on input gradients (smoothness or
  Lipschitz penalties, physics-informed residuals with respect to the input).

```{literalinclude} gradient_penalty.py
:language: python
```
