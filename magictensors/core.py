import torch
from torch import nn
# from koila import LazyTensor as KoilaTensor

class _MagicBehavior:
    FUNC_BEHAVIOR = 'func'
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __iter__(self):
        return self.key, self.value

    @staticmethod
    def make_func_behavior(orig_func, new_func):
        return _MagicBehavior(key=FUNC_BEHAVIOR, value=(orig_func, new_func))


class MagicTensor(torch.Tensor):
    def __init__(self, data, _behaviors=None, **kwargs):
        super().__init__(data, **kwargs)
        self._behaviors = _behaviors or set()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        behaviors_list = a._behaviors if hasattr(a, '_behaviors') else set() for a in args
        for arg_behaviors in behaviors_list:
            for behavior in arg_behaviors:
                key, value = behavior
                if key == _MagicBehavior.FUNC_BEHAVIOR:
                    orig_func, new_func = value
                    if orig_func == func:
                        func = new_func
                else:
                    raise NotImplementedError

        ret = super().__torch_function__(func, types, args, kwargs)
        final_behaviors = sum(behaviors_list, set())
        return MagicTensor(ret, _behaviors=final_behaviors)