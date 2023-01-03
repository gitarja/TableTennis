import numpy as np
from scipy.interpolate import interp1d
from numpy import array
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def extrap1d(interpolator, default_value=False):
    xs = interpolator.x
    ys = interpolator.y

    kernel = DotProduct() + WhiteKernel()


    def pointwise(x):
        if x < xs[0] or x > xs[-1]:
            if default_value:
                return np.NaN
            model = GaussianProcessRegressor(kernel=kernel,
                                             random_state=0, normalize_y=True, n_restarts_optimizer=2).fit(
                xs.reshape(-1, 1), ys.reshape(-1, 1))
            return model.predict(x.reshape(1, -1))[0, 0]
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike