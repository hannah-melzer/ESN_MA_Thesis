#from functools import partial
import numpy as np
from scipy.fft import dctn
from esn_dev.utils import _fromfile, normalize
from PIL import Image 
from scipy.signal import convolve

def make_operation(spec):
    """
    Creates an `Operation` from an operation spec.
    Each operation accepts a 2D input and outputs a vector:

      op(img) -> vec

    Possible operations specs are e.g.:

      * Resampled pixels:
        {"type":"pixel", "size": (3,3), "factor":1.}

      * Random projection:
          {"type":"random_weights", "input_size":10, "hidden_size":20,
           "factor": 1.}

      * Convolutions:
        {"type":"conv", "size": (4,4), # kernel size
         "kernel": kernel_type,        # either "gauss"/"random"
         "factor": 1.}

      * Gradient:
        {"type":"gradient", "factor": 1.}

      * Discrete Cosine Transform:
        {"type":"dct", "size": (n,n)  # pick first n coefficients in each dimension,
         "factor":0.1}

    Every operation spec must contain at least a 'type' determining the kind of
    operation and a 'factor' that is applied to the operation before outputing
    the result (realized through a 'ScaleOp')
    """
    optype = spec["type"]
    if optype == "pixels":
        op = PixelsOp(spec["size"])
    elif optype == "random_weights":
        op = RandWeightsOp(spec["input_size"], spec["hidden_size"])
    elif optype == "conv":
        op = ConvOp(spec["size"], spec["kernel"])
    elif optype == "gradient":
        op = GradientOp()
    elif optype == "vorticity":
        op = VorticityOp()        
    elif optype == "dct":
        op = DCTOp(spec["size"])
    else:
        raise ValueError(f"Unknown input map spec: {spec['type']}")
    #  TODO: normalize all of it? such that all outputs are between (-1,1)? # 
    return ScaleOp(spec["factor"], op)


def rescale(mapih, factors):
    ops = [ScaleOp(f,op.op) for (f,op) in zip(factors, mapih.ops)]
    return InputMap(ops)


class Operation:

    @classmethod
    def fromfile(cls, filename):
        return _fromfile(filename)

    @classmethod
    def fromspec(self, spec):
        return make_operation(spec)


class InputMap(Operation):
    """
    A callable object, that can be called with a 2D input image.  The
    `InputMap` is composed of a number of `Operation`s.  Each `Operation` again
    takes an image as input and outputs a vector.  Possible operations include
    convolutions, random maps, resize, etc. For a full list of operations and
    their specifications see `make_operation`.

    Params:
        specs: list of dicts with that each specify an `operation`

    Returns:
        A function that can be called with a 2D array and that outputs
        a 1D array (concatenated output of each op).
    """
    def __init__(self, xs):
        if isinstance(xs[0], dict):
            self.ops = [make_operation(spec) for spec in xs]
        elif isinstance(xs[0], ScaleOp):
            self.ops = xs
        else:
            raise ValueError("'xs' must either be a list of 'ScaleOp's or 'dict's!")

    def __call__(self, img):
        return np.concatenate([op(img) for op in self.ops], axis=0)

    def output_size(self, input_shape):
        return sum([op.output_size(input_shape) for op in self.ops])
    

    
def normalize_minmax(x):
    min_x, max_x = np.min(x), np.max(x)
    if max_x - min_x == 0:
        return np.zeros_like(x)
    return 2 * (x - min_x) / (max_x - min_x) - 1  # range [-1, 1]



class PixelsOp(Operation):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        with Image.fromarray(img) as im:
            im_resized = np.asarray(im.resize(self.size[::-1],resample=Image.BICUBIC))
        
        return im_resized.reshape(-1)
        # out = im_resized.reshape(-1)
        # return normalize_minmax(out)
    

    def output_size(self, input_shape):
        return self.size[0] * self.size[1]

    def output_shape(self, input_shape):
        return (self.size[0], self.size[1])



class RandWeightsOp(Operation):
    def __init__(self, input_size, hidden_size):
        self.isize = input_size
        self.hsize = hidden_size
        self.Wih = np.random.uniform(-1, 1, (self.hsize, self.isize))
        self.Wih = self.Wih / np.abs(self.Wih.sum(axis=1)).max()
        #self.bh  = np.random.uniform(-1, 1, (self.hsize,))

    def __call__(self, img):
        #Wih, bh = self.Wih, self.bh
        Wih = self.Wih
        return Wih.dot(img.reshape(-1)) #+ bh
        # flat = img.reshape(-1)
        # proj = self.Wih.dot(flat)
        # return normalize_minmax(proj)


    def output_size(self, input_shape):
        return self.hsize

    def output_shape(self, input_shape):
        raise NotImplementedError


class ScaleOp(Operation):
    def __init__(self, factor, op):
        self.factor = factor
        self.op = op

    def __call__(self, img):
        return self.factor * self.op(img)

    def output_size(self, input_shape):
        return self.op.output_size(input_shape)

    def output_shape(self, input_shape):
        return self.op.output_shape(input_shape)
    

# class GradientOp(Operation):
#     def __call__(self, img):
#         gx, gy = np.gradient(img)
#         grad_mag = np.sqrt(gx**2 + gy**2)
#         grad_mag /= (np.std(grad_mag) + 1e-8)
#         return grad_mag.reshape(-1)

class GradientOp(Operation):
    def __call__(self, img):
        x = np.gradient(img)
        x = np.concatenate(x).reshape(-1)
        return x #normalize(x)*2-1 
        # x = np.gradient(img)
        # out = np.concatenate(x).reshape(-1)
        # return normalize_minmax(out)

    def output_size(self, input_shape):
        s = self.output_shape(input_shape)
        return s[0] * s[1]

    def output_shape(self, input_shape):
        return (input_shape[0]*2, input_shape[1])


# class VorticityOp(Operation):
#     def __call__(self, img):
#         # Compute gradients
#         u, v = np.gradient(img)
        
#         # Compute vorticity using axis arguments to keep shape consistent
#         dv_dx = np.gradient(v, axis=1)
#         du_dy = np.gradient(u, axis=0)
        
#         vorticity = dv_dx - du_dy
        
#         # Subtract mean to center around zero
#         vorticity -= np.mean(vorticity)
        
#         # Assert to be sure shapes match
#         assert vorticity.shape == img.shape, f"Expected {img.shape}, got {vorticity.shape}"
        
#         return vorticity.reshape(-1)


class VorticityOp(Operation):
    def __call__(self, img):
        
        u, v = np.gradient(img)
        ux, uy = np.gradient(u)
        vx, vy = np.gradient(v)
        vorticity = ux - vy
        vorticity -= np.mean(vorticity)
        return vorticity.reshape(-1)
#         # return np.sqrt(img.size)*vorticity.reshape(-1)
#         # vorticity *= np.sqrt(img.size)
#         # return normalize_minmax(vorticity.reshape(-1))
    

    def output_size(self, input_shape):
        s = self.output_shape(input_shape)
        return s[0] * s[1]

    def output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
    

class ConvOp(Operation):
    def __init__(self, size, kernel, padding="same"):
        self.size = size
        self.name = kernel
        self.kernel = get_kernel(size, kernel)[np.newaxis,np.newaxis,:,:]
        self.padding = padding

    def __call__(self, img):
        img = np.expand_dims(img, axis=(0,1))
        out = convolve(img, self.kernel, mode=self.padding)
        return out.reshape(-1)
        # return normalize_minmax(out.reshape(-1))


    def output_size(self, input_shape):
        s = self.output_shape(input_shape)
        return s[0] * s[1]

    def output_shape(self, input_shape):
        p = self.padding
        if p == "same":
            return input_shape
        elif p == "valid":
            (m,n) = self.size
            return (input_shape[0]-m+1, input_shape[1]-n+1)
        else:
            raise ValueError(f"'padding' must be either 'valid' or 'same' - got: {p}")


class DCTOp(Operation):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        #Two dimensional discrete cosine transform
        #of image. Keep first (nk1,nk2) components 
        Fkk = dctn(img, type=2, workers=-1,norm='forward')
        Fkk = Fkk[:self.size[0],:self.size[1]]
        return Fkk.reshape(-1)
        # return normalize_minmax(Fkk.reshape(-1))


    def output_size(self, input_shape):
        return self.size[0] * self.size[1]

    def output_shape(self, input_shape):
        return self.size

def get_kernel(kernel_shape, kernel_type):
    if kernel_type == "mean":
        kernel = _mean_kernel(kernel_shape)
    elif kernel_type == "random":
        kernel = _random_kernel(kernel_shape)
    elif kernel_type == "gauss":
        kernel = _gauss_kernel(kernel_shape)
    else:
        raise NotImplementedError(f"Unkown kernel type `{kernel_type}`")
    return kernel


def _mean_kernel(kernel_shape):
    return np.ones(kernel_shape) / (kernel_shape[0] * kernel_shape[1])


def _random_kernel(kernel_shape):
    kernel = np.random.uniform(size=kernel_shape, low=-1, high=1)
    kernel = 2*kernel/np.abs(kernel).sum()
    return kernel

def _gauss_kernel(kernel_shape):
    ysize, xsize = kernel_shape
    yy = np.linspace(-ysize / 2., ysize / 2., ysize)
    xx = np.linspace(-xsize / 2., xsize / 2., xsize)
    sigma = min(kernel_shape) / 6.
    gaussian = np.exp(-(xx[:, None]**2 + yy[None, :]**2) / (2 * sigma**2))
    norm = np.sum(gaussian)  # L1-norm is 1
    gaussian = (1. / norm) * gaussian
    return gaussian
