# Pytorch Tensor Methods
```python

tensor.shape
>>> torch.Size([3, 3])

tensor.size()
>>> torch.Size([3, 3])

tensor.dtype()
>>> torch.float32

# Best to use since torch default is torch.float32 and numpy default is float64
tensor = torch.from_numpy(np_arr)
tensor = tensor.to(torch.float32)

tensor.view(10, -1)

```