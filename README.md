# KerasTensorPP
Implement keras-core ops in as methods of a class for easier method chaining.

## Example

```python
from keras_core import ops
from keras_tensor_pp import KerasTensorPP

(KerasTensorPP(ops.arange(10))
    .clip(4, 6)
    .cumsum()
    .log()
)
# <KerasTensorPP<tf.Tensor: shape=(10,), dtype=float64, numpy=
# array([1.38629436, 2.07944154, 2.48490665, 2.77258872, 2.99573227,
#       3.21887582, 3.4339872 , 3.61091791, 3.76120012, 3.8918203 ])>>
```
