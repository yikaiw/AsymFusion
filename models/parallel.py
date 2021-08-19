import torch.nn as nn


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class ModuleIndivParallel(nn.Module):
    def __init__(self, module, num_parallel, **kwargs):
        super(ModuleIndivParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'module_' + str(i), module(**kwargs))

    def forward(self, x_parallel):
        return [getattr(self, 'module_' + str(i))(x) for i, x in enumerate(x_parallel)]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]


class ShareBatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(ShareBatchNorm2dParallel, self).__init__()
        self.bn_0 = nn.BatchNorm2d(num_features)

    def forward(self, x_parallel):
        return [self.bn_0(x) for x in x_parallel]
