
import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
print(torch.cuda.device_count())

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

# 在PyTorch中，每个数组都有一个设备（device）， 我们通常将其称为环境（context）。 默认情况下，所有变量和相关的计算都分配给CPU。
# 无论何时我们要对多个项进行操作， 它们都必须在同一个设备上。
x = torch.tensor([1, 2, 3])
print(x.device)

