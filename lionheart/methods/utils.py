import torch
import types
import functools

def isint(i):
    return isinstance(i, int)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if (
            type(obj) == list
            or type(obj) == torch.fx.immutable_collections.immutable_list
        ):
            return obj.__getitem__(int(attr))
        else:
            if type(obj) == tuple:
                return obj.__getitem__(int(attr))
            else:
                return getattr(obj, attr)

    if isinstance(attr, types.BuiltinFunctionType):
        return None
    else:
        return functools.reduce(_getattr, [obj] + attr.split("."))

def rsetattr(obj, attr, val):
    def _rsetattr(subobj, attr, val, pre_hist=[]):
        nonlocal obj
        pre, _, post = attr.rpartition(".")
        immutable_map = None
        # If the object is immutable, we need to make a temporary mutable copy of it.
        if isinstance(subobj, (tuple, torch.fx.immutable_collections.immutable_list)):
            immutable_map = (tuple(pre_hist), type(subobj))
            subobj = list(subobj)
            if len(pre_hist) == 1:
                if isint(pre_hist[0]):
                    obj[int(pre_hist[0])] = list(obj[int(pre_hist[0])])
                else:
                    setarr(obj, pre_hist[0], list(obj[pre_hist[0]]))
            else:
                setattr(
                    rgetattr(obj, ".".join(pre_hist[:-1])),
                    pre_hist[-1],
                    list(rgetattr(obj, ".".join(pre_hist))),
                )

        if pre:
            pre_hist.append(pre)
            _rsetattr(rgetattr(subobj, pre), post, val, pre_hist=pre_hist)
        else:
            if type(subobj) == slice:
                if attr == "start":
                    s = slice(val, subobj.stop, subobj.step)
                elif attr == "stop":
                    s = slice(subobj.start, val, subobj.step)
                elif attr == "step":
                    s = slice(subobj.start, subobj.stop, val)

                if pre_hist != []:
                    if len(pre_hist) == 1:
                        if isint(pre_hist[0]):
                            obj[int(pre_hist[0])] = s
                        else:
                            rsetattr(obj, pre_hist[0], s)
                    else:
                        setattr(rgetattr(obj, ".".join(pre_hist[:-1])), pre_hist[-1], s)
                else:
                    obj = s
            elif all(i.isdigit() for i in attr):
                try:
                    subobj[int(attr)] = val
                except:
                    subobj = list(obj)
                    subobj[int(attr)] = val
                if pre_hist != []:
                    if len(pre_hist) == 1:
                        if isint(pre_hist[0]):
                            obj[int(pre_hist[0])] = subobj
                        else:
                            setattr(obj, pre_hist[0], subobj)
                    else:
                        setattr(
                            rgetattr(obj, ".".join(pre_hist[:-1])), pre_hist[-1], subobj
                        )
                else:
                    obj = subobj
            else:
                if pre_hist != []:
                    setattr(rgetattr(obj, ".".join(pre_hist)), attr, val)
                else:
                    setattr(obj, attr, val)

        if immutable_map is not None:
            if len(immutable_map[0]) == 1:
                if isint(immutable_map[0][0]):
                    obj[int(immutable_map[0][0])] = immutable_map[1](
                        obj[int(immutable_map[0][0])]
                    )
                else:
                    setarr(
                        obj,
                        immutable_map[0][0],
                        immutable_map[1](obj[immutable_map[0][0]]),
                    )
            else:
                setattr(
                    rgetattr(obj, ".".join(immutable_map[0][:-1])),
                    immutable_map[0][-1],
                    immutable_map[1](rgetattr(obj, ".".join(immutable_map[0]))),
                )

    _rsetattr(obj, attr, val, [])
    return obj

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
