from assimulo.explicit_ode import *
import numpy as N
import scipy.linalg as SL
import scipy.optimize as so
import matplotlib.pyplot as plt

class BDF_2(Explicit_ODE):
    """
    Explicit Euler.
    """

    # order: 1 - Explicit Euler, 2 - BDF2, 3 - BDF3, 4 - BDF4
    def __init__(self, model, order = 4, tol = 1.e-6, maxit = 100, maxsteps = 100000):
        super().__init__(model)
        self.method_order = order
        self.Tol = tol
        self.maxit = maxit
        self.maxsteps = maxsteps


    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        if opts["output_list"] == None:
            raise Explicit_ODE_Exception('BDF-2 is a fixed step-size method. Provide' \
                                         ' the number of communication points.')
        if self.method_order < 1 or self.method_order > 4:
            raise Explicit_ODE_Exception('Method order must be between 1 and 4 (inclusive)')
        
        self.h = N.diff(opts["output_list"])[0]
        tlist = []
        ylist = []
        
        # T = [t_n, t_nm1, t_nm2, t_nm3]
        T = [t, t, t, t]
        Y = [y, y, y, y]
        for i in range(self.maxsteps):
            if T[0] >= tf:
                break
            if i == 0 or self.method_order == 1:  # initial step
                t_np1, y_np1 = self.step_EE(T[0], Y[0])
            elif i == 1 or self.method_order == 2:
                t_np1, y_np1 = self.step_BDF2(T[:2], Y[:2])
            elif i == 2 or self.method_order == 3:
                t_np1, y_np1 = self.step_BDF3(T[:3], Y[:3])
            elif self.method_order == 4:
                t_np1, y_np1 = self.step_BDF4(T, Y)
            T = [t_np1] + T[:-1]
            Y = [y_np1] + Y[:-1]
            tlist.append(T[0])
            ylist.append(Y[0])
            self.h = min(self.h, N.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        #self.h = N.diff(opts["output_list"])[0]
        
        return 3, tlist, ylist
    
    def step_EE(self, t, y):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        f = self.problem.rhs
        h = self.h
        return t + h, y + h*f(t, y)
        
    def step_BDF2(self,T,Y):
        """
        BDF-2 with Fixed Point Iteration and Zero order predictor
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
        """
        alpha=[3./2.,-2.,1./2]
        f=self.problem.rhs
        h=self.h
        t_n,t_nm1=T
        y_n,y_nm1=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        for i in range(self.maxit):
            y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1)+h*f(t_np1,y_np1_i))/alpha[0]
            if SL.norm(y_np1_ip1-y_np1_i) < self.Tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)

    def step_BDFn(self, T, Y, n, alpha):
        """
        Performs a BDF step of order n
        """
        f = self.problem.rhs
        h = self.h
        # predictor
        t_np1 = T[0] + h
        y_np1_i = Y[0] # zero order predictor
        def g(y_np1):
            return alpha[0]*y_np1 + sum([alpha[i+1]*Y[i] for i in range(n)]) - h*f(t_np1, y_np1)
        y_np1 = so.fsolve(g, y_np1_i)
        return t_np1, y_np1

    def step_BDF3(self,T,Y):
        """
        BDF-3 with fsolve
        """
        alpha = [11/6, -18/6, 9/6, -2/6]
        return self.step_BDFn(T, Y, 3, alpha)

    def step_BDF4(self,T,Y):
        """
        BDF-4 with fsolve
        """
        alpha = [25/12, -48/12, 36/12, -16/12, 3/12]
        return self.step_BDFn(T, Y, 4, alpha)
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics: %s \n' % self.problem.name,        verbose)
        self.log_message(' Step-length          : %s '%(self.h), verbose)
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF2',                       verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)

def lambda_func(y1, y2, k = 10):
    return k*(N.sqrt(y1**2 + y2**2) - 1)/N.sqrt(y1**2 + y2**2)

#the right hand side of our problem
def rhs(t, y):
    return N.array([y[2], y[3], -y[0]*lambda_func(y[0], y[1]), -y[1]*lambda_func(y[0], y[1]) - 1])

#initial values. y[0] = x-position, y[1] = y-position, y[2] = x-velocity, y[3] = y-velocity
y0 = N.array([1.2, 0.0, 0.0, 0.0])
t0 = 0.0

#Assimulo stuff
model = Explicit_Problem(rhs, y0, t0)
model.name = 'Task 1'
sim = BDF_2(model, 1)
#sim.atol = 0.1
#sim.rtol = 0.1
#sim.maxord = 1
#simulation. Store result in t and y
tfinal = 20
n_steps = 1000
t, y = sim.simulate(tfinal, n_steps)

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
