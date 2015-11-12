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
    def __init__(self, order = 4, tol = 1.e-8, maxit = 100, maxsteps = 100000):
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
        
        t_nm1 = 0 # making sure these are defined for first iterations
        t_nm2 = 0
        y_nm1 = 0
        y_nm2 = 0
        for i in range(self.maxsteps):
            if t >= tf:
                break
            if i==0 or self.method_order == 1:  # initial step
                t_np1,y_np1 = self.step_EE(t,y)
            elif i == 1 or self.method_order == 2:
                t_np1, y_np1 = self.step_BDF2([t,t_nm1], [y,y_nm1])
            elif i == 2 or self.method_order == 3:
                t_np1, y_np1 = self.step_BDF3([t,t_nm1, t_nm2], [y,y_nm1, y_nm2])
            elif self.method_order == 4:
                t_np1, y_np1 = self.step_BDF4([t,t_nm1, t_nm2, t_nm3], [y,y_nm1, y_nm2, y_nm3])    
            t, t_nm1, t_nm2, t_nm3 = t_np1, t, t_nm1, t_nm2
            y, y_nm1, y_nm2, y_nm3 = y_np1, y, y_nm1, y_nm2
            tlist.append(t)
            ylist.append(y)
    
            self.h=min(self.h,N.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        self.h = N.diff(opts["output_list"])[0]
        
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

    def step_BDF3(self,T,Y):
        """
        BDF-3 with fsolve
        """
        alpha = [11/6, -18/6, 9/6, -2/6]
        f=self.problem.rhs
        h=self.h
        t_n,t_nm1,tnm2=T
        y_n,y_nm1,y_nm2=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        def g(y_np1):
            return alpha[0]*y_np1 + alpha[1]*y_n + alpha[2]*y_nm1 + alpha[3]*y_nm2 - h*f(t_np1, y_np1)
        y_np1_ip1 = so.fsolve(g, y_np1_i)
        return t_np1, y_np1_ip1

    def step_BDF4(self,T,Y):
        """
        BDF-4 with fsolve
        """
        alpha = [25/12, -48/12, 36/12, -16/12, 3/12]
        f=self.problem.rhs
        h=self.h
        t_n, t_nm1, tnm2, t_nm3 = T
        y_n, y_nm1, y_nm2, y_nm3 = Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        def g(y_np1):
            return alpha[0]*y_np1 + alpha[1]*y_n + alpha[2]*y_nm1 + alpha[3]*y_nm2 + alpha[4]*y_nm3 - h*f(t_np1, y_np1)
        y_np1_ip1 = so.fsolve(g, y_np1_i)
        return t_np1, y_np1_ip1

            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics: %s \n' % self.problem.name,        verbose)
        self.log_message(' Step-length          : %s '%(self.h), verbose)
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF2',                       verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
            

'''#TEST EXAMPLE
def pend(t,y):
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return N.array([y[1],-gl*N.sin(y[0])])

#Initial conditions
y0=N.array([0.5*N.pi,0.])    

#Specify an explicit problem
pend_mod=Explicit_Problem(pend, y0)
pend_mod.name='Nonlinear Pendulum'

#Define an explicit solver
exp_sim = BDF_2(pend_mod) #Create a BDF solver

#Simulate the problem
t,y = exp_sim.simulate(1.0, 10)

#Plot the result
P.plot(t,y)
P.show()'''

def lambda_func(y1, y2, k = 20):
    return k*(N.sqrt(y1**2 + y2**2) - 1)/N.sqrt(y1**2 + y2**2)

#the right hand side of our problem
def rhs(t, y):
    return N.array([y[2], y[3], -y[0]*lambda_func(y[0], y[1]), -y[1]*lambda_func(y[0], y[1]) - 1])

#initial values. y[0] = x-position, y[1] = y-position, y[2] = x-velocity, y[3] = y-velocity
y0 = N.array([1.4, 0.0, 0.0, 0.0])
t0 = 0.0

#Assimulo stuff
model = Explicit_Problem(rhs, y0, t0)
model.name = 'Task 1'
sim = BDF_2(model)
sim.method_order = 4
#sim.atol = 0.1
#sim.rtol = 0.1
#sim.maxord = 1
#sim.maxh = 0.01
#sim.minh = 0.01
#sim = CVode(model)
tfinal = 20
#simulation. Store result in t and y
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
