from itertools import chain
from functools import reduce
import torch
from torch import nn
# from koila import LazyTensor as KoilaTensor

class _MagicBehavior:
    FUNC_BEHAVIOR = 'func'
    def __init__(self, kind, value, has_unique_id, unique_id=None):
        self.kind = kind
        self.value = value
        self.has_unique_id = has_unique_id
        self.unique_id = unique_id

    def __iter__(self):
        return iter([self.kind, self.value])

    @staticmethod
    def make_func_behavior(orig_func, new_func):
        return _MagicBehavior(kind=_MagicBehavior.FUNC_BEHAVIOR, value=(orig_func, new_func), 
                              has_unique_id=True, unique_id=(_MagicBehavior.FUNC_BEHAVIOR, orig_func))

    
class _BehaviorPool:
    def __init__(self):
        self._unique_ids = {}
        self._non_uniques = set()
    
    def add(self, x):
        if x.has_unique_id:
            self._unique_ids[x.unique_id] = x
        else:
            self._non_uniques.add(x)
    
    def __or__(self, other):
        ret = _BehaviorPool()
        ret._unique_ids = {**self._unique_ids, **other._unique_ids}
        ret._non_uniques = self._non_uniques | other._non_uniques
        return ret
    
    def __iter__(self):
        return chain(iter(self._non_uniques), iter(self._unique_ids.values()))
    
    def __repr__(self):
        return str(set(iter(self)))
    
class MagicTensor(torch.Tensor):
    @staticmethod 
    def __new__(cls, x, *args, **kwargs):
        if '_behaviors' in kwargs:
            kwargs.pop('_behaviors')
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, data, _behaviors=None, **kwargs):
        self._behaviors = _behaviors or _BehaviorPool()
    
    def __repr__(self):
        return "MagicTensor\n" + torch.Tensor(self).__repr__()
    
    def override_function(self, orig_func, new_func):
        self._behaviors.add(_MagicBehavior.make_func_behavior(orig_func, new_func))
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func == cls.__repr__:
            return func(torch.Tensor(args[0]))
        if kwargs is None:
            kwargs = {}
        behaviors_list = [a._behaviors if hasattr(a, '_behaviors') else _BehaviorPool() for a in args]
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
        final_behaviors = reduce(lambda x, y: x | y, behaviors_list, _BehaviorPool())
        return MagicTensor(ret, _behaviors=final_behaviors)
