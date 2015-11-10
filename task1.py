import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from BDF2 import BDF_2

def lambda_func(y1, y2, k = 1):
    return k*(np.sqrt(y1**2 + y2**2) - 1)/np.sqrt(y1**2 + y2**2)

def rhs(t, y):
    return np.array([y[2], y[3], -y[0]*lambda_func(y[0], y[1]), -y[1]*lambda_func(y[0], y[1]) - 1])

y0 = np.array([1.0, 0.0, 0.0, -1.0])
t0 = 0.0

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Task 1'
sim = BDF_2(model)
#sim = CVode(model)
sim.maxord = 7
tfinal = 100.0
t, y = sim.simulate(tfinal)
#pl.plot(y[:, 0], y[:, 1])
#pl.show()
#sim.plot()
fig, ax = plt.subplots()
ax.plot(t, y[:, 0], label='x-pos')
ax.plot(t, y[:, 1], label='y-pos')
legend = ax.legend(loc='upper center', shadow=True)

plt.figure(1)
fig2, ax2 = plt.subplots()
ax2.plot(t, y[:, 2], label='x-vel')
ax2.plot(t, y[:, 3], label='y-vel')
legend = ax2.legend(loc='upper center', shadow=True)

plt.figure(1)
fig3, ax3 = plt.subplots()
ax3.plot(y[:, 0], y[:, 1], label='displacement')
legend = ax3.legend(loc='upper center', shadow=True)

#ax.plot(a, c, 'k--', label='Model length')
#ax.plot(a, d, 'k:', label='Data length')
#ax.plot(a, c+d, 'k', label='Total message length')

# Now add the legend with some customizations.
plt.show()
