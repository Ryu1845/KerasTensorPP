from inspect import signature
from textwrap import dedent

import keras_core

if __name__ == "__main__":
    with open("keras_tensor_pp.py", "r") as file:
        file.write(dedent("""
            from keras_core import ops

            class KerasTensorPP:
            """)
        )
        ops = [op for op in dir(keras_core.ops) if "__" not in op]
        for op in ops:
            sig = signature(eval(f"keras_core.{op}"))
            first_param_key, first_param = next(sig.parameters.items())
            params = sig.parameters.copy()
            params.pop(first_param_key)
            params["self"] = first_param.replace(name="self")
            sig = sig.replace(parameters=params.values())
            parameters = [param for param in sig.parameters.keys()]
            file.write(dedent(f"""
                def {op}{str(sig)}:
                    return ops.{op}({','.join(parameters))
                """)
