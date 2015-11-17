import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.solvers import ExplicitEuler
from assimulo.solvers import ImplicitEuler
from assimulo.solvers import LSODAR
from assimulo.solvers import ODASSL
from assimulo.solvers import Dopri5

#lambda(y1, y2, k)

def lambda_func(y1, y2, k = 10):
    return k*(np.sqrt(y1**2 + y2**2) - 1)/np.sqrt(y1**2 + y2**2)

#the right hand side of our problem
def rhs(t, y):
    return np.array([y[2], y[3], -y[0]*lambda_func(y[0], y[1]), -y[1]*lambda_func(y[0], y[1]) - 1])

#initial values. y[0] = x-position, y[1] = y-position, y[2] = x-velocity, y[3] = y-velocity
y0 = np.array([1.2, 0.0, 0.0, 0.0])
t0 = 0.0

#Assimulo stuff
model = Explicit_Problem(rhs, y0, t0)
model.name = 'Task 1'
sim = Dopri5(model)
#sim.atol = 1e-2
#sim.rtol = 1e-2
#sim.maxord = 1
#sim.maxh = 0.2
#sim.minh = 0.001
#sim = CVode(model)
#sim.discr = 'Adams'
tfinal = 20
#simulation. Store result in t and y
#tcor, ycor = sim.simulate(tfinal)
#sim.atol = 0.05
#sim.rtol = 0.05
t, y = sim.simulate(tfinal)

#Create plots. Three figures are created: one containing positional values as a function of time, one with velocities as a function of time and the last traces the pendulum's movement (x and y coordinates in cartesian coordinate)
fig, ax = plt.subplots()
ax.plot(t, y[:, 0], label='x-pos')
ax.plot(t, y[:, 1], label='y-pos')
legend = ax.legend(loc='upper center', shadow=True)
plt.grid()

plt.figure(1)
fig2, ax2 = plt.subplots()
ax2.plot(t, y[:, 2], label='x-vel')
ax2.plot(t, y[:, 3], label='y-vel')
legend = ax2.legend(loc='upper center', shadow=True)
plt.grid()

plt.figure(1)
fig3, ax3 = plt.subplots()
ax3.plot(y[:, 0], y[:, 1], label='displacement')
legend = ax3.legend(loc='upper center', shadow=True)
plt.grid()

# Now add the legend with some customizations.
plt.show()
