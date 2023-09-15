import keras_core

def keras_tensor_pp_wrapper(fn):
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        return KerasTensorPP.from_kt(ret) if not isinstance(ret, tuple) else tuple(KerasTensorPP.from_kt(r) for r in ret)
    return wrapper

KerasTensorPP = type("KerasTensorPP",(keras_core.KerasTensor,),{op:eval(f"keras_tensor_pp_wrapper(keras_core.ops.{op})") for op in dir(keras_core.ops) if "__" not in op})
KerasTensorPP.from_kt = classmethod(lambda cls, kt: cls(kt.shape, kt.dtype, kt.record_history, kt.name))
