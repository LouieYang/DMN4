import torch

def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))

def l2distance(x, y):
    """
    Input:
        x [.., c, M_x]
        y [.., c, M_y]
    Return:
        ret [.., M_x, M_y]
    """
    
    assert x.shape[:-2] == y.shape[:-2]
    prefix_shape = x.shape[:-2]

    c, M_x = x.shape[-2:]
    M_y = y.shape[-1]
    
    x = x.view(-1, c, M_x)
    y = y.view(-1, c, M_y)

    x_t = x.transpose(1, 2)
    x_t2 = x_t.pow(2.0).sum(-1, keepdim=True)
    y2 = y.pow(2.0).sum(1, keepdim=True)

    ret = x_t2 + y2 - 2.0 * x_t@y
    ret = ret.view(prefix_shape + (M_x, M_y))
    return ret

def batched_index_select(input_, dim, index):
    for ii in range(1, len(input_.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input_.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input_, dim, index)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module

class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn

