import numpy as np


def generateSurrogateData(t="random", n=1000):
    def logistic_map(r, x):

        return (r * x * (1 - x))

    if t == "uniform":
        data = np.random.uniform(size=(n,))
    elif t == "chaotic":
        r = 3.8
        x0 = .2
        chaotic_signal = [x0]
        for _ in range(n):
            x_next = logistic_map(r, chaotic_signal[-1])
            chaotic_signal.append(x_next)

        data =  np.asarray(chaotic_signal)
    else:
        data = np.random.randn(n)

    return np.expand_dims(data, axis=0)


def generateFourierPhaseSurrogate(v1, gaussian=True):
    # Step 1: Compute the Fourier transform of the data
    fft_data1 = np.fft.fft(v1)

    amplitude1 = np.abs(fft_data1)
    phase1 = np.angle(fft_data1)

    if gaussian:
        v2 = np.random.normal(loc=0, scale=1, size=len(v1))

        fft_data2 = np.fft.fft(v2)

        phase2 = np.angle(fft_data2)

        # Construct surrogate data
        surrogate_fft = amplitude1 * np.exp(1j * phase2)

    else:

        # Randomize phase
        phase_randomized = np.random.uniform(0, 2 * np.pi, len(phase1))

        # Construct surrogate data
        surrogate_fft = amplitude1 * np.exp(1j * phase_randomized)

    surrogate_data = np.fft.ifft(surrogate_fft).real

    return surrogate_data


def correlatedNoiseSurrogate(original_data, surrogates):
    """
    Return Fourier surrogates.

    Generate surrogates by Fourier transforming the :attr:`original_data`
    time series (assumed to be real valued), randomizing the phases and
    then applying an inverse Fourier transform. Correlated noise surrogates
    share their power spectrum and autocorrelation function with the
    original_data time series.

    The Fast Fourier transforms of all time series are cached to facilitate
    a faster generation of several surrogates for each time series. Hence,
    :meth:`clear_cache` has to be called before generating surrogates from
    a different set of time series!

    .. note::
       The amplitudes are not adjusted here, i.e., the
       individual amplitude distributions are not conserved!

    **Examples:**

    The power spectrum is conserved up to small numerical deviations:
    True

    However, the time series amplitude distributions differ:

    False

    :type original_data: 2D array [index, time]
    :arg original_data: The original time series.
    :rtype: 2D array [index, time]
    :return: The surrogate time series.
    """

    # #  Calculate FFT of original_data time series
    # #  The FFT of the original_data data has to be calculated only once,
    # #  so it is stored in self._original_data_fft.
    # if self._fft_cached:
    #     surrogates = self._original_data_fft
    # else:
    #     surrogates = np.fft.rfft(original_data, axis=1)
    #     self._original_data_fft = surrogates
    #     self._fft_cached = True

    #  Get shapes
    (N, n_time) = original_data.shape
    len_phase = surrogates.shape[1]

    #  Generate random phases uniformly distributed in the
    #  interval [0, 2*Pi]
    phases = np.random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

    #  Add random phases uniformly distributed in the interval [0, 2*Pi]
    surrogates = surrogates * np.exp(1j * phases)

    #  Calculate IFFT and take the real part, the remaining imaginary part
    #  is due to numerical errors.
    return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,
                                                     axis=1)))


def AAFTSurrogates(original_data, surrogates):
    """
    Return surrogates using the amplitude adjusted Fourier transform
    method.

    Reference: [Schreiber2000]_

    :type original_data: 2D array [index, time]
    :arg original_data: The original time series.
    :rtype: 2D array [index, time]
    :return: The surrogate time series.
    """
    #  Create sorted Gaussian reference series
    gaussian = np.random.randn(original_data.shape[0], original_data.shape[1])
    gaussian.sort(axis=1)

    #  Rescale data to Gaussian distribution
    ranks = original_data.argsort(axis=1).argsort(axis=1)
    rescaled_data = np.zeros(original_data.shape)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = gaussian[i, ranks[i, :]]

    #  Phase randomize rescaled data
    phase_randomized_data = correlatedNoiseSurrogate(rescaled_data, surrogates)

    #  Rescale back to amplitude distribution of original data
    sorted_original = original_data.copy()
    sorted_original.sort(axis=1)

    ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = sorted_original[i, ranks[i, :]]

    return rescaled_data


def generateIAAFTData(v1, nmax_iter=1000):
    # Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
    """
    Return surrogates using the iteratively refined amplitude adjusted
    Fourier transform method.

    A set of AAFT surrogates (:meth:`AAFT_surrogates`) is iteratively
    refined to produce a closer match of both amplitude distribution and
    power spectrum of surrogate and original data.

    Reference: [Schreiber2000]_

    :type original_data: 2D array [index, time]
    :arg original_data: The original time series.
    :arg int n_iterations: Number of iterations / refinement steps
    :arg str output: Type of surrogate to return. "true_amplitudes":
        surrogates with correct amplitude distribution, "true_spectrum":
        surrogates with correct power spectrum, "both": return both outputs
        of the algorithm.
    :rtype: 2D array [index, time]
    :return: The surrogate time series.
    """
    #  Get size of dimensions
    n_time = v1.shape[1]

    #  Get Fourier transform of original data with caching

    ori_fourier_transform = np.fft.rfft(v1, axis=1)

    #  Get Fourier amplitudes
    original_fourier_amps = np.abs(ori_fourier_transform)

    #  Get sorted copy of original data
    sorted_original = v1.copy()
    sorted_original.sort(axis=1)

    #  Get starting point / initial conditions for R surrogates
    # (see [Schreiber2000]_)
    R = AAFTSurrogates(v1, ori_fourier_transform)

    #  Start iteration
    for i in range(nmax_iter):
        #  Get Fourier phases of R surrogate
        r_fft = np.fft.rfft(R, axis=1)
        r_phases = r_fft / np.abs(r_fft)

        #  Transform back, replacing the actual amplitudes by the desired
        #  ones, but keeping the phases exp(iψ(i)
        s = np.fft.irfft(original_fourier_amps * r_phases, n=n_time,
                         axis=1)

        #  Rescale to desired amplitude distribution
        ranks = s.argsort(axis=1).argsort(axis=1)

        for j in range(v1.shape[0]):
            R[j, :] = sorted_original[j, ranks[j, :]]



    return R


if __name__ == '__main__':
    v1 = generateSurrogateData(t="uniform", n=1000)
    v2 = generateIAAFTData(v1, nmax_iter=1000)
    import matplotlib.pyplot as plt

    plt.plot(v1[0])
    plt.plot(v2[0])
    plt.show()
