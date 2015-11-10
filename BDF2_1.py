from assimulo.explicit_ode import *
import numpy as N
import scipy.linalg as SL
import scipy.optimize as so

class BDF_2(Explicit_ODE):
    """
    Explicit Euler.
    """
    Tol=1.e-8
    maxit=100
    maxsteps = 100000
    
    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        if opts["output_list"] == None:
            raise Explicit_ODE_Exception('BDF-2 is a fixed step-size method. Provide' \
                                         ' the number of communication points.')
        
        self.h = N.diff(opts["output_list"])[0]
        
        tlist = []
        ylist = []
        
        t_nm1 = 0 # making sure these are defined for first iterations
        y_nm1 = 0
        print('hej')
        for i in range(self.maxsteps):
            if t >= tf:
                break
            if i==0:  # initial step
                t_np1,y_np1 = self.step_EE(t,y)
                print('hej2')
            elif i == 1:
                t_np1, y_np1 = self.step_BDF2([t,t_nm1], [y,y_nm1])
                print('hej3')
            else:   
                t_np1, y_np1 = self.step_BDF3([t,t_nm1, t_nm2], [y,y_nm1, y_nm2])
                print('hej4')
            t, t_nm1, t_nm2 = t_np1, t, t_nm1
            y, y_nm1, y_nm2 = y_np1, y, y_nm1
            print('t = ', t, ', y = ', y)
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
        BDF-2 with Fixed Point Iteration and Zero order predictor
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
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
            return y_np1 - 18*y_n/11 + 9*y_nm1/11 - 2*y_nm2/11 - 6*h*f(t_np1, y_np1)
        y_np1_ip1 = so.fsolve(g, y_np1_i)
        return t_np1, y_np1_ip1

            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics: %s \n' % self.problem.name,        verbose)
        self.log_message(' Step-length          : %s '%(self.h), verbose)
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF2',                       verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
            

#TEST EXAMPLE
def pend(t,y):
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return N.array([y[1],-gl*N.sin(y[0])])

#Initial conditions
y0=N.array([2.*N.pi,0.])    

#Specify an explicit problem
pend_mod=Explicit_Problem(pend, y0)
pend_mod.name='Nonlinear Pendulum'

#Define an explicit solver
exp_sim = BDF_2(pend_mod) #Create a BDF solver

#Simulate the problem
t,y = exp_sim.simulate(1.0, 10)

#Plot the result
P.plot(t,y)
P.show()
