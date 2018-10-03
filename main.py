"""
paper: https://www.sciencedirect.com/science/article/pii/S0378779605002415
"""
import numpy as np
import matplotlib.pylab as plt

import signalz
import padasip as pa

np.random.seed(101)

N = 15000
n = 20
n_skip = 10000
dummy = False

# if dummy:
#     d = signalz.levy_flight(N, alpha=1.8, beta=0., sigma=1., position=0)
# else:
#     filename = "data/DAT_ASCII_EURUSD_M1_201809.csv"
#     d = np.loadtxt(open(filename, "rb"), delimiter=";", skiprows=1, usecols=(1,))


q = signalz.random_steps(500, steps_count=40, distribution="standard", std=3, mean=0)
N = len(q)
d = q + signalz.gaussian_white_noise(N, offset=0, std=0.1)



x = pa.input_from_history(d, n)[:-1]
d = d[n:]


filters = [
    {"filter": pa.filters.FilterNLMS(n=n, mu=1., w="zeros")},
    {"filter": pa.filters.FilterRLS(n=n, mu=0.95, w="random")},
    {"filter": pa.filters.FilterLMS(n=n, mu=0.1, w="random")},
]


# for f in filters:
#     y, e, w = f["filter"].run(d, x)
#     elbnd = pa.detection.ELBND(w, e, function="max")


y, e, w = filters[1]["filter"].run(d, x)
elbnd = pa.detection.ELBND(w, e, function="max")






plt.subplot(311)
plt.plot(d[n_skip:])
plt.plot(y[n_skip:])
plt.plot(q[n_skip:], "k")


plt.subplot(312)
plt.plot(elbnd[n_skip:])

plt.subplot(313)
plt.plot(np.abs(np.diff(d)[n_skip:]))

plt.show()

