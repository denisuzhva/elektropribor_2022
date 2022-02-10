import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.optimize import linprog



## Data load

data_path = "./data/synthetic_data.txt"
data = np.loadtxt(data_path)[:, 3]
data /= data.max()

## Data process

n = data.size
fq = 1./1.
t = np.linspace(0.0, n*fq, n, endpoint=False)

n_f = n//2
data_f = fft(data)
t_f = fftfreq(n, fq)[:n_f]
data_f_norm = 2.0/n * np.abs(data_f[0:n_f])

## CS

cr = 2
m = n_f // cr
a_mat = np.random.normal(0, 1/cr, size=(m, n_f))
y_f = a_mat.dot(data_f_norm)

c_arr = np.ones(n_f)
res = linprog(c_arr, A_eq=a_mat, b_eq=y_f)['x']
data_rec = np.array(res)

plt.plot(data_f_norm)
plt.plot(data_rec)
plt.show()

np.save("./data/compressed.npy", data_f_norm)


