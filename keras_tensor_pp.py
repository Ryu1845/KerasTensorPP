
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

    def abs(self):
        return self._wrap(ops.abs(self._tensor, ))

    def absolute(self):
        return self._wrap(ops.absolute(self._tensor, ))

    def add(self, x2):
        return self._wrap(ops.add(self._tensor, x2))

    def all(self, axis=None, keepdims=False):
        return self._wrap(ops.all(self._tensor, axis, keepdims))

    def amax(self, axis=None, keepdims=False):
        return self._wrap(ops.amax(self._tensor, axis, keepdims))

    def amin(self, axis=None, keepdims=False):
        return self._wrap(ops.amin(self._tensor, axis, keepdims))

    def any(self, axis=None, keepdims=False):
        return self._wrap(ops.any(self._tensor, axis, keepdims))

    def append(self, x2, axis=None):
        return self._wrap(ops.append(self._tensor, x2, axis))

    def arange(self, stop=None, step=1, dtype=None):
        return self._wrap(ops.arange(self._tensor, stop, step, dtype))

    def arccos(self):
        return self._wrap(ops.arccos(self._tensor, ))

    def arcsin(self):
        return self._wrap(ops.arcsin(self._tensor, ))

    def arcsinh(self):
        return self._wrap(ops.arcsinh(self._tensor, ))

    def arctan(self):
        return self._wrap(ops.arctan(self._tensor, ))

    def arctan2(self, x2):
        return self._wrap(ops.arctan2(self._tensor, x2))

    def arctanh(self):
        return self._wrap(ops.arctanh(self._tensor, ))

    def argmax(self, axis=None):
        return self._wrap(ops.argmax(self._tensor, axis))

    def argmin(self, axis=None):
        return self._wrap(ops.argmin(self._tensor, axis))

    def argsort(self, axis=-1):
        return self._wrap(ops.argsort(self._tensor, axis))

    def array(self, dtype=None):
        return self._wrap(ops.array(self._tensor, dtype))

    def average(self, axis=None, weights=None):
        return self._wrap(ops.average(self._tensor, axis, weights))

    def average_pool(self, pool_size, strides=None, padding='valid', data_format=None):
        return self._wrap(ops.average_pool(self._tensor, pool_size, strides, padding, data_format))

    def binary_crossentropy(self, output, from_logits=False):
        return self._wrap(ops.binary_crossentropy(self._tensor, output, from_logits))

    def bincount(self, weights=None, minlength=0):
        return self._wrap(ops.bincount(self._tensor, weights, minlength))

    def broadcast_to(self, shape):
        return self._wrap(ops.broadcast_to(self._tensor, shape))

    def cast(self, dtype):
        return self._wrap(ops.cast(self._tensor, dtype))

    def categorical_crossentropy(self, output, from_logits=False, axis=-1):
        return self._wrap(ops.categorical_crossentropy(self._tensor, output, from_logits, axis))

    def ceil(self):
        return self._wrap(ops.ceil(self._tensor, ))

    def clip(self, x_min, x_max):
        return self._wrap(ops.clip(self._tensor, x_min, x_max))

    def concatenate(self, axis=0):
        return self._wrap(ops.concatenate(self._tensor, axis))

    def cond(self, true_fn, false_fn):
        return self._wrap(ops.cond(self._tensor, true_fn, false_fn))

    def conj(self):
        return self._wrap(ops.conj(self._tensor, ))

    def conjugate(self):
        return self._wrap(ops.conjugate(self._tensor, ))

    def conv(self, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
        return self._wrap(ops.conv(self._tensor, kernel, strides, padding, data_format, dilation_rate))

    def conv_transpose(self, kernel, strides, padding='valid', output_padding=None, data_format=None, dilation_rate=1):
        return self._wrap(ops.conv_transpose(self._tensor, kernel, strides, padding, output_padding, data_format, dilation_rate))

    def convert_to_tensor(self, dtype=None):
        return self._wrap(ops.convert_to_tensor(self._tensor, dtype))

    def copy(self):
        return self._wrap(ops.copy(self._tensor, ))

    def cos(self):
        return self._wrap(ops.cos(self._tensor, ))

    def cosh(self):
        return self._wrap(ops.cosh(self._tensor, ))

    def count_nonzero(self, axis=None):
        return self._wrap(ops.count_nonzero(self._tensor, axis))

    def cross(self, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
        return self._wrap(ops.cross(self._tensor, x2, axisa, axisb, axisc, axis))

    def cumprod(self, axis=None):
        return self._wrap(ops.cumprod(self._tensor, axis))

    def cumsum(self, axis=None):
        return self._wrap(ops.cumsum(self._tensor, axis))

    def depthwise_conv(self, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
        return self._wrap(ops.depthwise_conv(self._tensor, kernel, strides, padding, data_format, dilation_rate))

    def diag(self, k=0):
        return self._wrap(ops.diag(self._tensor, k))

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return self._wrap(ops.diagonal(self._tensor, offset, axis1, axis2))

    def digitize(self, bins):
        return self._wrap(ops.digitize(self._tensor, bins))

    def divide(self, x2):
        return self._wrap(ops.divide(self._tensor, x2))

    def dot(self, x2):
        return self._wrap(ops.dot(self._tensor, x2))

    def einsum(self, *operands):
        return self._wrap(ops.einsum(self._tensor, operands))

    def elu(self, alpha=1.0):
        return self._wrap(ops.elu(self._tensor, alpha))

    def empty(self, dtype='float32'):
        return self._wrap(ops.empty(self._tensor, dtype))

    def equal(self, x2):
        return self._wrap(ops.equal(self._tensor, x2))

    def exp(self):
        return self._wrap(ops.exp(self._tensor, ))

    def expand_dims(self, axis):
        return self._wrap(ops.expand_dims(self._tensor, axis))

    def expm1(self):
        return self._wrap(ops.expm1(self._tensor, ))

    def extract_sequences(self, sequence_length, sequence_stride):
        return self._wrap(ops.extract_sequences(self._tensor, sequence_length, sequence_stride))

    def eye(self, M=None, k=0, dtype='float32'):
        return self._wrap(ops.eye(self._tensor, M, k, dtype))

    def fft(self):
        return self._wrap(ops.fft(self._tensor, ))

    def fft2(self):
        return self._wrap(ops.fft2(self._tensor, ))

    def flip(self, axis=None):
        return self._wrap(ops.flip(self._tensor, axis))

    def floor(self):
        return self._wrap(ops.floor(self._tensor, ))

    def floor_divide(self, x2):
        return self._wrap(ops.floor_divide(self._tensor, x2))

    def fori_loop(self, upper, body_fun, init_val):
        return self._wrap(ops.fori_loop(self._tensor, upper, body_fun, init_val))

    def full(self, fill_value, dtype=None):
        return self._wrap(ops.full(self._tensor, fill_value, dtype))

    def full_like(self, fill_value, dtype=None):
        return self._wrap(ops.full_like(self._tensor, fill_value, dtype))

    def gelu(self, approximate=True):
        return self._wrap(ops.gelu(self._tensor, approximate))

    def get_item(self, key):
        return self._wrap(ops.get_item(self._tensor, key))

    def greater(self, x2):
        return self._wrap(ops.greater(self._tensor, x2))

    def greater_equal(self, x2):
        return self._wrap(ops.greater_equal(self._tensor, x2))

    def hard_sigmoid(self):
        return self._wrap(ops.hard_sigmoid(self._tensor, ))

    def hstack(self):
        return self._wrap(ops.hstack(self._tensor, ))

    def identity(self, dtype='float32'):
        return self._wrap(ops.identity(self._tensor, dtype))

    def imag(self):
        return self._wrap(ops.imag(self._tensor, ))

    def in_top_k(self, predictions, k):
        return self._wrap(ops.in_top_k(self._tensor, predictions, k))

    def irfft(self, fft_length=None):
        return self._wrap(ops.irfft(self._tensor, fft_length))

    def isclose(self, x2):
        return self._wrap(ops.isclose(self._tensor, x2))

    def isfinite(self):
        return self._wrap(ops.isfinite(self._tensor, ))

    def isinf(self):
        return self._wrap(ops.isinf(self._tensor, ))

    def isnan(self):
        return self._wrap(ops.isnan(self._tensor, ))

    def istft(self, sequence_length, sequence_stride, fft_length, length=None, window='hann', center=True):
        return self._wrap(ops.istft(self._tensor, sequence_length, sequence_stride, fft_length, length, window, center))

    def leaky_relu(self, negative_slope=0.2):
        return self._wrap(ops.leaky_relu(self._tensor, negative_slope))

    def less(self, x2):
        return self._wrap(ops.less(self._tensor, x2))

    def less_equal(self, x2):
        return self._wrap(ops.less_equal(self._tensor, x2))

    def linspace(self, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
        return self._wrap(ops.linspace(self._tensor, stop, num, endpoint, retstep, dtype, axis))

    def log(self):
        return self._wrap(ops.log(self._tensor, ))

    def log10(self):
        return self._wrap(ops.log10(self._tensor, ))

    def log1p(self):
        return self._wrap(ops.log1p(self._tensor, ))

    def log2(self):
        return self._wrap(ops.log2(self._tensor, ))

    def log_sigmoid(self):
        return self._wrap(ops.log_sigmoid(self._tensor, ))

    def log_softmax(self, axis=-1):
        return self._wrap(ops.log_softmax(self._tensor, axis))

    def logaddexp(self, x2):
        return self._wrap(ops.logaddexp(self._tensor, x2))

    def logical_and(self, x2):
        return self._wrap(ops.logical_and(self._tensor, x2))

    def logical_not(self):
        return self._wrap(ops.logical_not(self._tensor, ))

    def logical_or(self, x2):
        return self._wrap(ops.logical_or(self._tensor, x2))

    def logical_xor(self, x2):
        return self._wrap(ops.logical_xor(self._tensor, x2))

    def logspace(self, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
        return self._wrap(ops.logspace(self._tensor, stop, num, endpoint, base, dtype, axis))

    def logsumexp(self, axis=None, keepdims=False):
        return self._wrap(ops.logsumexp(self._tensor, axis, keepdims))

    def matmul(self, x2):
        return self._wrap(ops.matmul(self._tensor, x2))

    def max(self, axis=None, keepdims=False, initial=None):
        return self._wrap(ops.max(self._tensor, axis, keepdims, initial))

    def max_pool(self, pool_size, strides=None, padding='valid', data_format=None):
        return self._wrap(ops.max_pool(self._tensor, pool_size, strides, padding, data_format))

    def maximum(self, x2):
        return self._wrap(ops.maximum(self._tensor, x2))

    def mean(self, axis=None, keepdims=False):
        return self._wrap(ops.mean(self._tensor, axis, keepdims))

    def meshgrid(*self, indexing='xy'):
        return self._wrap(ops.meshgrid(self._tensor, indexing))

    def min(self, axis=None, keepdims=False, initial=None):
        return self._wrap(ops.min(self._tensor, axis, keepdims, initial))

    def minimum(self, x2):
        return self._wrap(ops.minimum(self._tensor, x2))

    def mod(self, x2):
        return self._wrap(ops.mod(self._tensor, x2))

    def moveaxis(self, source, destination):
        return self._wrap(ops.moveaxis(self._tensor, source, destination))

    def multi_hot(self, num_tokens, axis=-1, dtype=None):
        return self._wrap(ops.multi_hot(self._tensor, num_tokens, axis, dtype))

    def multiply(self, x2):
        return self._wrap(ops.multiply(self._tensor, x2))

    def nan_to_num(self):
        return self._wrap(ops.nan_to_num(self._tensor, ))

    def ndim(self):
        return self._wrap(ops.ndim(self._tensor, ))

    def negative(self):
        return self._wrap(ops.negative(self._tensor, ))

    def nonzero(self):
        return self._wrap(ops.nonzero(self._tensor, ))

    def not_equal(self, x2):
        return self._wrap(ops.not_equal(self._tensor, x2))

    def one_hot(self, num_classes, axis=-1, dtype=None):
        return self._wrap(ops.one_hot(self._tensor, num_classes, axis, dtype))

    def ones(self, dtype='float32'):
        return self._wrap(ops.ones(self._tensor, dtype))

    def ones_like(self, dtype=None):
        return self._wrap(ops.ones_like(self._tensor, dtype))

    def outer(self, x2):
        return self._wrap(ops.outer(self._tensor, x2))

    def pad(self, pad_width, mode='constant'):
        return self._wrap(ops.pad(self._tensor, pad_width, mode))

    def power(self, x2):
        return self._wrap(ops.power(self._tensor, x2))

    def prod(self, axis=None, keepdims=False, dtype=None):
        return self._wrap(ops.prod(self._tensor, axis, keepdims, dtype))

    def qr(self, mode='reduced'):
        return self._wrap(ops.qr(self._tensor, mode))

    def ravel(self):
        return self._wrap(ops.ravel(self._tensor, ))

    def real(self):
        return self._wrap(ops.real(self._tensor, ))

    def reciprocal(self):
        return self._wrap(ops.reciprocal(self._tensor, ))

    def relu(self):
        return self._wrap(ops.relu(self._tensor, ))

    def relu6(self):
        return self._wrap(ops.relu6(self._tensor, ))

    def repeat(self, repeats, axis=None):
        return self._wrap(ops.repeat(self._tensor, repeats, axis))

    def reshape(self, new_shape):
        return self._wrap(ops.reshape(self._tensor, new_shape))

    def rfft(self, fft_length=None):
        return self._wrap(ops.rfft(self._tensor, fft_length))

    def roll(self, shift, axis=None):
        return self._wrap(ops.roll(self._tensor, shift, axis))

    def round(self, decimals=0):
        return self._wrap(ops.round(self._tensor, decimals))

    def rsqrt(self):
        return self._wrap(ops.rsqrt(self._tensor, ))

    def scatter(self, values, shape):
        return self._wrap(ops.scatter(self._tensor, values, shape))

    def scatter_update(self, indices, updates):
        return self._wrap(ops.scatter_update(self._tensor, indices, updates))

    def segment_max(self, segment_ids, num_segments=None, sorted=False):
        return self._wrap(ops.segment_max(self._tensor, segment_ids, num_segments, sorted))

    def segment_sum(self, segment_ids, num_segments=None, sorted=False):
        return self._wrap(ops.segment_sum(self._tensor, segment_ids, num_segments, sorted))

    def selu(self):
        return self._wrap(ops.selu(self._tensor, ))

    def separable_conv(self, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
        return self._wrap(ops.separable_conv(self._tensor, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate))

    def shape(self):
        return self._wrap(ops.shape(self._tensor, ))

    def sigmoid(self):
        return self._wrap(ops.sigmoid(self._tensor, ))

    def sign(self):
        return self._wrap(ops.sign(self._tensor, ))

    def silu(self):
        return self._wrap(ops.silu(self._tensor, ))

    def sin(self):
        return self._wrap(ops.sin(self._tensor, ))

    def sinh(self):
        return self._wrap(ops.sinh(self._tensor, ))

    def size(self):
        return self._wrap(ops.size(self._tensor, ))

    def slice(self, start_indices, shape):
        return self._wrap(ops.slice(self._tensor, start_indices, shape))

    def slice_update(self, start_indices, updates):
        return self._wrap(ops.slice_update(self._tensor, start_indices, updates))

    def softmax(self, axis=-1):
        return self._wrap(ops.softmax(self._tensor, axis))

    def softplus(self):
        return self._wrap(ops.softplus(self._tensor, ))

    def softsign(self):
        return self._wrap(ops.softsign(self._tensor, ))

    def sort(self, axis=-1):
        return self._wrap(ops.sort(self._tensor, axis))

    def sparse_categorical_crossentropy(self, output, from_logits=False, axis=-1):
        return self._wrap(ops.sparse_categorical_crossentropy(self._tensor, output, from_logits, axis))

    def split(self, indices_or_sections, axis=0):
        return self._wrap(ops.split(self._tensor, indices_or_sections, axis))

    def sqrt(self):
        return self._wrap(ops.sqrt(self._tensor, ))

    def square(self):
        return self._wrap(ops.square(self._tensor, ))

    def squeeze(self, axis=None):
        return self._wrap(ops.squeeze(self._tensor, axis))

    def stack(self, axis=0):
        return self._wrap(ops.stack(self._tensor, axis))

    def std(self, axis=None, keepdims=False):
        return self._wrap(ops.std(self._tensor, axis, keepdims))

    def stft(self, sequence_length, sequence_stride, fft_length, window='hann', center=True):
        return self._wrap(ops.stft(self._tensor, sequence_length, sequence_stride, fft_length, window, center))

    def stop_gradient(self):
        return self._wrap(ops.stop_gradient(self._tensor, ))

    def subtract(self, x2):
        return self._wrap(ops.subtract(self._tensor, x2))

    def sum(self, axis=None, keepdims=False):
        return self._wrap(ops.sum(self._tensor, axis, keepdims))

    def swapaxes(self, axis1, axis2):
        return self._wrap(ops.swapaxes(self._tensor, axis1, axis2))

    def swish(self):
        return self._wrap(ops.swish(self._tensor, ))

    def take(self, indices, axis=None):
        return self._wrap(ops.take(self._tensor, indices, axis))

    def take_along_axis(self, indices, axis=None):
        return self._wrap(ops.take_along_axis(self._tensor, indices, axis))

    def tan(self):
        return self._wrap(ops.tan(self._tensor, ))

    def tanh(self):
        return self._wrap(ops.tanh(self._tensor, ))

    def tensordot(self, x2, axes=2):
        return self._wrap(ops.tensordot(self._tensor, x2, axes))

    def tile(self, repeats):
        return self._wrap(ops.tile(self._tensor, repeats))

    def top_k(self, k, sorted=True):
        return self._wrap(ops.top_k(self._tensor, k, sorted))

    def trace(self, offset=0, axis1=0, axis2=1):
        return self._wrap(ops.trace(self._tensor, offset, axis1, axis2))

    def transpose(self, axes=None):
        return self._wrap(ops.transpose(self._tensor, axes))

    def tri(self, M=None, k=0, dtype='float32'):
        return self._wrap(ops.tri(self._tensor, M, k, dtype))

    def tril(self, k=0):
        return self._wrap(ops.tril(self._tensor, k))

    def triu(self, k=0):
        return self._wrap(ops.triu(self._tensor, k))

    def true_divide(self, x2):
        return self._wrap(ops.true_divide(self._tensor, x2))

    def unstack(self, num=None, axis=0):
        return self._wrap(ops.unstack(self._tensor, num, axis))

    def var(self, axis=None, keepdims=False):
        return self._wrap(ops.var(self._tensor, axis, keepdims))

    def vdot(self, x2):
        return self._wrap(ops.vdot(self._tensor, x2))

    def vstack(self):
        return self._wrap(ops.vstack(self._tensor, ))

    def where(self, x1=None, x2=None):
        return self._wrap(ops.where(self._tensor, x1, x2))

    def while_loop(self, body, loop_vars, maximum_iterations=None):
        return self._wrap(ops.while_loop(self._tensor, body, loop_vars, maximum_iterations))

    def zeros(self, dtype='float32'):
        return self._wrap(ops.zeros(self._tensor, dtype))

    def zeros_like(self, dtype=None):
        return self._wrap(ops.zeros_like(self._tensor, dtype))
