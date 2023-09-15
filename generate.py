from inspect import signature, Signature
from textwrap import dedent, indent

import keras_core

if __name__ == "__main__":
    with open("keras_tensor_pp.py", "w") as file:
        file.write(dedent("""
            from keras_core import ops

            class KerasTensorPP:
                def __init__(self, tensor):
                    self._tensor = tensor

                @classmethod
                def _wrap(cls, res):
                    if isinstance(res, tuple):
                        return tuple(cls._wrap(r) for r in res)
                    return cls(res)

                def __repr__(self):
                    return f"<KerasTensorPP{self._tensor!r}>"
            """)
        )
        ops = [op for op in dir(keras_core.ops) if not any(pattern in op for pattern in ("__", "image", "nn", "numpy"))]
        for op in ops:
            sig = signature(eval(f"keras_core.ops.{op}"))
            first_param_key, first_param = next(iter(sig.parameters.items()))
            params = sig.parameters.copy()
            params.pop(first_param_key)
            params = list(params.values())
            params.insert(0, first_param.replace(name="self"))
            sig = Signature(params, __validate_parameters__=False)
            parameters = [param for param in sig.parameters.keys()]
            function = indent(dedent(f"""
                def {op}{str(sig)}:
                    return self._wrap(ops.{op}(self._tensor, {', '.join(parameters[1:])}))
                """), "    ")
            file.write(function)

