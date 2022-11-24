from itertools import chain
from functools import reduce
import torch
from torch import nn
# from koila import LazyTensor as KoilaTensor


### General Utils
class _Item:
    def __init__(self, kind, value, has_unique_id, unique_id=None):
        self.kind = kind
        self.value = value
        self.has_unique_id = has_unique_id
        self.unique_id = unique_id


class _Pool:
    def __init__(self):
        self._unique_ids = {}
        self._non_uniques = set()
    
    def add(self, x):
        if x.has_unique_id:
            self._unique_ids[x.unique_id] = x
        else:
            self._non_uniques.add(x)
    
    def __or__(self, other):
        ret = _Pool()
        ret._unique_ids = {**self._unique_ids, **other._unique_ids}
        ret._non_uniques = self._non_uniques | other._non_uniques
        return ret
    
    def __iter__(self):
        return chain(iter(self._non_uniques), iter(self._unique_ids.values()))
    
    def __repr__(self):
        return str(set(iter(self)))
    
    def get(self, id_name, default_value=None):
        return self._unique_ids[id_name] if id_name in self._unique_ids.keys() else default_value


### Computation Graphs
class _ComputationGraph(_Item):
    UNIQUE_ID = '_ComputationGraph'
    def __init__(self, func, args, kwargs):
        args = list(map(self._get_graph, args))
        kwargs = {k: self._get_graph(v) for k, v in kwargs.items()}
        super().__init__(kind=_ComputationGraph.UNIQUE_ID, value=(func, args, kwargs), 
                         has_unique_id=True, unique_id=_ComputationGraph.UNIQUE_ID)
        
    @staticmethod
    def _get_graph(x):
        if is_magic_tensor(x):
            return x._states.get(_ComputationGraph.UNIQUE_ID)
        else:
            return None
        

### Magic Behaviors 
class _MagicBehavior(_Item):
    FUNC_BEHAVIOR = 'func'
    TRACK_HISTORY = 'track_history'
    
    def __iter__(self):
        return iter([self.kind, self.value])

    def __repr__(self):
        value = self.value
        if self.kind == _MagicBehavior.FUNC_BEHAVIOR:
            orig_func, new_func = value
            return f'function override: {orig_func} -> {new_func}' 
        elif self.kind == _MagicBehavior.TRACK_HISTORY:
            if self.value:
                return 'track history: on'
            else: 
                return ''
        else:
            raise NotImplementedError
    
    @staticmethod
    def make_func_behavior(orig_func, new_func):
        return _MagicBehavior(kind=_MagicBehavior.FUNC_BEHAVIOR, value=(orig_func, new_func), 
                              has_unique_id=True, unique_id=(_MagicBehavior.FUNC_BEHAVIOR, orig_func))
    
    @staticmethod
    def make_history_tracker(on):
        return _MagicBehavior(kind=_MagicBehavior.TRACK_HISTORY, value=on, 
                              has_unique_id=True, unique_id=_MagicBehavior.TRACK_HISTORY)
    
### Magic Tensor
class MagicTensor(torch.Tensor):
    @staticmethod 
    def __new__(cls, x, *args, **kwargs):
        if '_behaviors' in kwargs:
            kwargs.pop('_behaviors')
        if '_states' in kwargs:
            kwargs.pop('_states')
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, data, _behaviors=None, _states=None, **kwargs):
        self._behaviors = _behaviors or _Pool()
        self._states = _states or _Pool()
    
    def __repr__(self):
        return "MagicTensor\n" + torch.Tensor(self).__repr__()
    
    def show_magics(self):
        for behavior in self._behaviors:
            s = str(behavior)
            if s:
                print(behavior)
    
    def override_function(self, orig_func, new_func):
        self._behaviors.add(_MagicBehavior.make_func_behavior(orig_func, new_func))
    
    def track_history(self, on=True):
        self._behaviors.add(_MagicBehavior.make_history_tracker(on=on))
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func == cls.__repr__:
            return func(torch.Tensor(args[0]))
        if kwargs is None:
            kwargs = {}
        behaviors_list = [a._behaviors if hasattr(a, '_behaviors') else _Pool() for a in args]
        final_states = _Pool()
        
        for arg_behaviors in behaviors_list:
            for behavior in arg_behaviors:
                key, value = behavior
                if key == _MagicBehavior.FUNC_BEHAVIOR:
                    orig_func, new_func = value
                    if orig_func == func:
                        func = new_func
                elif key == _MagicBehavior.TRACK_HISTORY:
                    on = value
                    if on:
                        final_states.add(_ComputationGraph(func, args, kwargs))
                else:
                    raise NotImplementedError

        ret = super().__torch_function__(func, types, args, kwargs)
        final_behaviors = reduce(lambda x, y: x | y, behaviors_list, _Pool())
        return MagicTensor(ret, _behaviors=final_behaviors, _states=final_states)
    
# GUY TODO: we use a workaround, but there is a deep cause that functions like .shape dont work..
    @property
    def shape(self):
        return torch.Tensor(self).shape

    def size(self):
        return torch.Tensor(self).size()

    def dim(self):
        return torch.Tensor(self).dim()

    @property
    def ndim(self):
        return torch.Tensor(self).ndim

    def item(self):
        return torch.Tensor(self).item()
    
# more helper functions
def is_magic_tensor(x):
    return isinstance(x, MagicTensor)
