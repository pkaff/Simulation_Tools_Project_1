from assimulo.explicit_ode import *
import numpy as N
import scipy.linalg as SL

class BDF_3(Explicit_ODE):
	"""
	Explicit Euler.
	"""
	Tol=1.e-8
	maxit=100
		
	def integrate(self, t, y, tf, dt):
		"""
		_integrates (t,y) values until t > tf
		"""
		if dt <= 0.0:
			raise Explicit_ODE_Exception('BDF-2 is a fixed step-size method. Provide' \
										 ' the number of communication points.')
		
		self.h = dt

		for i in range(1, self.maxsteps):
			if t >= tf:
				break
			if i==1:  # initial step
				t_np1,y_np1 = self.step_EE(t,y)
				yield t_np1, y_np1
				t_np2,y_np2 = self.step_EE(t_np1,y_np1)
			else:	
				t_np1, y_np1 = self.step_BDF3([t,t_nm1, t_nm2], [y,y_nm1, y_nm2])
			t, t_nm1, t_nm2 = t_np1, t, t_nm1
			y, y_nm1, y_nm2 = y_np1, y, y_nm1
			yield t,y
			self.h=min(self.h,N.abs(tf-t))
		else:
			raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
	
	def step_EE(self, t, y):
		"""
		This calculates the next step in the integration with explicit Euler.
		"""
		f = self.f
		h = self.h
		return t + h, y + h*f(t, y) 

	def step_BDF3(self,T,Y):
		"""
		BDF-2 with Fixed Point Iteration and Zero order predictor
		
		alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
		alpha=[3/2,-2,1/2]
		"""
		alpha = [11/6, -18/6, 9/6, -2/6]
		f=self.f
		h=self.h
		t_n,t_nm1,tnm2=T
		y_n,y_nm1,y_nm2=Y
		# predictor
		t_np1=t_n+h
		y_np1_i=y_n   # zero order predictor
		# corrector with fixed point iteration
				
		for i in range(self.maxit):
			y_np1_ip1 = (-(alpha[1]*y_n + alpha[2]*y_nm1 + alpha[3]*y_nm2) + h*f(t_np1, y_np1_i))/alpha[0]
			#y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1)+h*f(t_np1,y_np1_i))/alpha[0]
			if SL.norm(y_np1_ip1-y_np1_i) < self.Tol:
				return t_np1,y_np1_ip1
			y_np1_i=y_np1_ip1
		else:
			raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)
	
if __name__ == '__main__':	
	#Define the rhs
	def f(t,y):
		ydot = -y[0]
		return N.array([ydot])
	
	#Define an Assimulo problem
	exp_mod = Explicit_Problem()
	exp_mod.f = f
	exp_mod.problem_name = 'Simple BDF-2 Example'

	def pend(t,y):
		#g=9.81    l=0.7134354980239037
		gl=13.7503671
		return N.array([y[1],-gl*N.sin(y[0])])
	pend_mod=Explicit_Problem()
	pend_mod.f=pend
	pend_mod.problem_name='Nonlinear Pendulum'
	pend_mod.y0=N.array([2.*N.pi,0.])
	#Define an explicit solver
	y0 = 4.0 #Initial conditions
	exp_sim = BDF_2(exp_mod)#,y0) #Create a BDF solver
