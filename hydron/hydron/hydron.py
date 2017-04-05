import os,sys
dir = os.path.dirname(__file__)
import threading
import numpy as np
from .config import Scale

def average(q):
    return 0.5*(q[1:]+q[:-1])


def delta(q, order = 1):
    if (order == 1):
        return (q[1:]-q[:-1])
    if (order == 2):
        return (q[2:]+q[:-2]-2*q[1:-1])

    
def smooth(x, d, mode = 'gaussian'):
    if (mode == 'gaussian'):
        from scipy.ndimage.filters import gaussian_filter1d
        return gaussian_filter1d(x,d,mode='wrap')
    if (mode == 'box'):
        return np.convolve(x, np.ones(d)/d, mode='same')


def load(filename):
    try:
        import pickle, dill
        return pickle.load(open(filename,"rb"))
    except:
        print("can't load file")

        
def save(obj, filename):
    try:
        import pickle, dill
        pickle.dump(obj,open(filename, "wb"))
    except:
        print("can't save object")

        

    
        
        
class Data(object):
    def __init__(self, scale=Scale()):
        self.x = []
        self.q = [[],[],[]]
        self.time = []
        self.set_scale(scale)
    
    
    def set_scale(self, scale):
        self.scale = scale
        return self
    
    def add(self,time,x,q):
        for i in range(0,3):
            self.q[i] += [q[i]]
        self.x += [x]
        self.time += [time] 
            
    def n(self, scale=0):
        if (scale == 0):
            scale = self.scale.n
        return np.array(self.q[0])*scale

    def u(self, scale=0):
        if (scale == 0):
            scale = self.scale.u
        return np.array(self.q[1])/self.n(1)*scale
    
    def T(self, scale=0):
        if (scale == 0):
            scale = self.scale.T
        return (self.q[2]/self.n(1)-self.u(1)**2/2)*(self.scale.gamma-1)*scale
    
    def tau(self, scale=0):
        if (scale == 0):
            scale = self.scale.t
        return self.time[-1]*scale
    
    def s(self, scale=0):
        if (scale == 0):
            scale = self.scale.x
        return np.array(self.x)*scale
    
    def L(self, scale=0):
        if (scale == 0):
            scale = self.scale.x
        return self.x[-1]*scale
    
    def __del__(self):
        return
        
    
        
class Mainloop(threading.Thread):
    def __init__(self,solver,dt,tau,data,verbose):
        threading.Thread.__init__(self)
        self.solver = solver
        self.data = data
        self.dt = dt
        self.tau = tau

        self.verbose = verbose
        self.wait_event = threading.Event()
        
        
    def wait(self):
        self.wait_event.wait()
        
        
    def run(self):
        self.status = 'in progress'
        if self.verbose:
            import time
            systime = time.time()
            
        t0 = self.solver.time
        while (self.solver.time - t0) < self.tau:
            self.solver.step(self.dt)
            #try:
            #    self.solver.step(self.dt)
            #except:
            #    self.status = 'unexpected error!'
            #    if self.verbose:
            #        print '\n'+self.status
            #    break
            if (type(self.data) == Data):
                self.data.add(self.solver.time, np.copy(self.solver.x),
                              [np.copy(self.solver.q[0]),
                               np.copy(self.solver.q[1]),
                               np.copy(self.solver.q[2])])
            
            if self.verbose:
                print('\r' + self.status + '...\t%d%%' % ((self.solver.time - t0)/self.tau*100),)
               
        self.status = 'done!'
        self.wait_event.set()
        
        if self.verbose:
            print('\n'+self.status)
            print('Elapsed time: {:.2f}'.format(time.time()-systime)+' seconds')
            
    def __del__(self):
        return
    
        
class Term(object):
    def __init__(self, *args):
        if (len(args) == 1):
            if (type(args[0]) == tuple):
                args = args[0]
                     
        if (type(args[0]) == str):
            self.name = args[0]
            
        if (len(args) > 1):
            self.value = args[1]
                
        if (len(args) > 2):
            if (type(args[2]) == Scale):
                self.scale = args[2]
                
        self.cfl = 0            

class Diffusion(Term):
    def __init__(self, *args):
        Term.__init__(self, *args)
        
        if (self.name == 'Spitzer'):
            if (type(self.value) == float):
                kappa0 = self.value
            
            kappa0 /= self.scale.x**2/self.scale.T**3.5*self.scale.rhoe/self.scale.t 
            self.get = lambda T: kappa0*T**2.5
            

    def step(self, solver, dt):
        Di = average(self.get(solver.T))
        Fi = solver.Ai*Di*delta(solver.T)/solver.dx
        solver.q[2][1:-1] += dt*delta(Fi)/solver.dxi/solver.A[1:-1]
        self.cfl = np.max(Di/solver.dx**2)
                

                
        
class Source(Term):
    def __init__(self, *args):
        Term.__init__(self, *args)
        
        if (self.name == 'heat'):
            scale = self.scale.rhoe/self.scale.t
            
            if callable(self.value):
                self.get = lambda s,t: self.value(s*self.scale.x,t*self.scale.t)/scale

            if (type(self.value) == float):
                h0 = self.value/scale
                self.get = lambda *s: h0

            def step(solver, dt):
                solver.q[2] += dt*self.get(solver.x,solver.time)

            self.step = step  

                
        if (self.name == 'radiation'):
            if (type(self.value) == str):
                radloss = np.load(dir+'\\radloss\\'+self.value)
                logR0 = radloss['rate']
                logT0 = radloss['temperature']
                    
                logR0 -= np.log10(self.scale.rhoe/self.scale.n**2/self.scale.t)
                logT0 -= np.log10(self.scale.T)
                            
                self.get = lambda T: 10**(np.interp(np.log10(T), logT0, logR0))
                    
            def step(solver, dt):
                solver.q[2] -= dt*solver.q[0]**2*self.get(solver.T)

            self.step = step
                
                
        if (self.name == 'force'):        
            scale = self.scale.a 


            if (type(self.value) == float):
                a0 = self.value/scale
                self.get = lambda ex: np.sum([a0*ex[-1]],0)
                    
            if (type(self.value) == np.ndarray):
                shape = self.value.shape
                a0 = np.copy(self.value)/scale
                self.get = lambda ex: np.sum([a0[i]*ex[i] for i in range(0,shape[0])],0)


            def step(solver, dt):
                solver.q[2] += dt*solver.q[1]*self.get(solver.ex)
                solver.q[1] += dt*solver.q[0]*self.get(solver.ex)

            self.step = step 
                

        
class Solver(object):
    def __init__(self,X,n,u,T,config,diffusion=('None',),sources=[],A=[1]):
        self.set_config(config)
        
        self.set_diffusion(diffusion)
        self.set_sources(sources)
        
        self.set_geometry(X,A)
        self.set_initial_values(n,u,T)
        
        
        self.time = 0
    

    def interface(self,p,type = ''):
        if (type == ''):
            type = self.config.itype
        if (type == 'Roe'):
            return ((p*np.sqrt(self.q[0]))[:-1]+(p*np.sqrt(self.q[0]))[1:])/(np.sqrt(self.q[0][:-1])+np.sqrt(self.q[0][1:]))
        if (type == 'average'):
            return average(p)
        
        return p

    
    def d_dx(self,p,order=1):
        dx1 = self.dx[:-1]
        dx2 = self.dx[1:]
        dp = delta(p)
        dp1 = dp[:-1]
        dp2 = dp[1:]
        
        if (order == 1):
            return ((dp2*dx1**2+dp1*dx2**2)/
                    ((dx1+dx2)*dx1*dx2))
        if (order == 2):
            return (2*(dp2*dx1-dp1*dx2)/
                    ((dx1+dx2)*dx1*dx2))
    
   
    def set_config(self, config):
        self.config = config
        self.scale = config.scale
    
        
    def set_geometry(self,X, A=False):
        if (len(X.shape) == 1):
            self.ndim = 1
            self.nx = X.shape[0]
        
        if (len(X.shape) == 2):
            self.ndim = X.shape[0]
            self.nx = X.shape[1]
        
        self.idx = range(0,self.nx)
        
        self.X = np.reshape(X,(self.ndim,self.nx))/self.scale.x
        self.Xi = np.zeros((self.ndim,self.nx-1),dtype=np.float32)

        
        self.update_x()
        self.x0 = np.copy(self.x)
        
        if (len(A) > 1):
            self.A = np.copy(A)/self.scale.x**2
        else:
            self.A = np.ones(self.nx,dtype=np.float32)*A
        
        self.A0 = np.copy(self.A)
        self.Ai = average(self.A)
        
        
    def update_x(self):
        for i in range(0,self.ndim):
            self.Xi[i] = average(self.X[i])
        
        self.dx = np.sqrt(np.sum([(delta(self.X[i]))**2 for i in range(0,self.ndim)],0))
        self.dxi = average(self.dx)
        
        self.ex = [delta(self.Xi[i])/self.dxi for i in range(0,self.ndim)]
        self.exi = np.array([delta(self.X[i])/self.dx for i in range(0,self.ndim)])

        for i in range(0,self.ndim):
            self.ex[i] = np.append(np.append(self.ex[i][0],self.ex[i]),self.ex[i][-1]) 
        self.ex = np.array(self.ex)
        
        self.x = np.cumsum(np.append([0],self.dx))
        self.xi = np.cumsum(np.append([self.dx[0]/2],self.dxi))
        self.L = self.x[-1]
        
        
    def update_A(self):
        self.A = np.interp(self.x,self.x0,self.A0)
        self.Ai = average(self.A)
    
    
    def set_initial_values(self,n,u,T):
        rho = n/self.scale.n
        rhou = rho*u/self.scale.u
        rhoe = 0.5*rhou**2/rho + rho*T/self.scale.T/(self.scale.gamma-1)
        
        self.w = np.zeros_like(u)
        self.wi = average(self.w)
        
        self.q = [rho,rhou,rhoe]
        self.q0 = [np.copy(rho),np.copy(rhou),np.copy(rhoe)]
        self.T0 = np.copy(T/self.scale.T)
        
        self.set_boundary()
        return self
    
    
    def set_diffusion(self,diffusion):
        self.diffusion = Diffusion(diffusion+(self.scale,))

        
    def set_sources(self,sources):
        self.sources = []
        for source in sources:
            self.sources += [Source(source + (self.scale,))]
        
    def set_boundary(self):
        if (self.config.btype == 'mirror'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[1,-2]]*(-1)**i
        
        if (self.config.btype == 'continuous'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[1,-2]]
        
        if (self.config.btype == 'periodic'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[-2,1]]
            
        if (self.config.btype == 'constant'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q0[i][[0,-1]]
    
        if (self.config.btype == 'stationary'):
            self.q[2][[0,-1]] = (self.q[2][[1,-2]]-0.5*self.q[1][[1,-2]]**2/self.q[0][[1,-2]]-
                                 self.q[0][[1,-2]]*self.sources[0].get(self.exi[:,[0,-1]])*
                                 self.dx[[0,-1]]*np.array([1,-1])/(self.scale.gamma-1))
            
            
            self.q[0][[0,-1]] = self.q[2][[0,-1]]/(self.T0[[0,-1]]/(self.scale.gamma-1))
            self.q[1][[0,-1]] = 0
            
            

            
    
    def set_cells(self):
        self.dq = [delta(self.q[i]*self.A) for i in range(0,3)]
        
        self.u = self.q[1]/self.q[0]
        self.ui = self.interface(self.u)
        
        self.T = (self.q[2]/self.q[0]-self.u**2/2)*(self.scale.gamma-1)
        self.p = self.q[0]*self.T
        self.c_s = np.sqrt(self.scale.gamma*self.T)         
    
        self.cfl = np.max([np.max((np.abs(self.ui)/self.dx)),
                    np.max((self.c_s[1:-1]/self.dxi)),self.diffusion.cfl])
        

    def get_substep(self, dt):
        return np.ceil(self.cfl*dt/self.config.cfl_lim).astype(int)
                
    def set_fluxes(self):
        if False:
            #self.w[1:-1] =  smooth(self.u[1:-1],3)
            #self.w =  self.u*1

            dT = self.d_dx(self.T)

            self.w[1:-1] = delta(self.T,2)*dT/(dT**2+1e-2)*0.1
            #self.w = smooth(self.w,self.nx/20.)

            self.wi = average(self.w)  
            F = [self.q[0]*(self.u-self.w),self.q[1]*(self.u-self.w)+self.p,self.q[2]*(self.u-self.w)+self.p*self.u] 
            
        else:    
            F = [self.q[0]*self.u,(self.q[1]*self.u+self.p),(self.q[2]*self.u+self.p*self.u)]  
            
            
        self.Fi = [self.Ai*average(F[i]) for i in range(0,3)]  ####
        
        
    
    def set_correction(self):
        ui = self.ui
        wi = self.wi

        
        if (self.config.ctype == 'upwind'):
            for i in range(0,3):
                self.Fi[i] -= 0.5*np.abs(ui)*self.dq[i]
                
        if (self.config.ctype == 'Riemann'):
            one = np.ones(self.nx-1,dtype=np.float32)
            zero = np.zeros(self.nx-1,dtype=np.float32)
            gamma = self.scale.gamma
            
            c_sl = self.c_s[:-1]   ###
            c_sr = self.c_s[1:]   ###

            self.dq[1]-=ui*self.dq[0]
            self.dq[2]-=ui*self.dq[1]+ui**2/2*self.dq[0]

            lam = [ui-c_sl,ui,ui+c_sr]
            detR = c_sl**2/(gamma-1) + c_sr**2/(gamma-1)
            
            R = [[one,one,one],
                 [-c_sl,zero,c_sr],
                 [c_sl**2/(gamma-1),zero,c_sr**2/(gamma-1)]]

            Ri = [[zero,-c_sl/(gamma-1),one],
                  [c_sl**2/(gamma-1)+c_sr**2/(gamma-1),zero,-2*one],
                  [zero,c_sr/(gamma-1),one]]
            
            ai = [np.sum([Ri[i][j]/detR*self.dq[j] for j in range(0,3)],0) for i in range(0,3)]
            Wi = [[R[i][j]*ai[j] for j in range(0,3)] for i in range(0,3)]

            for i in range(0,3):       
                self.Fi[i] -= 0.5*np.sum([np.abs(lam[j])*Wi[i][j] for j in range(0,3)],0)     
          
                
    
    def advect(self,dt):
        for i in range(0,3):
            self.q[i][1:-1] -= dt*delta(self.Fi[i])/self.dxi/self.A[1:-1]
            
    def step(self,dt):
        self.set_cells()
        n_s = self.get_substep(dt)
        self.set_fluxes()
        self.set_correction()

        self.advect(dt/n_s)
        self.q[1][1:-1] += dt/n_s*self.p[1:-1]*self.d_dx(self.A)/self.A[1:-1]
        self.q[2][1:-1] += dt/n_s*self.p[1:-1]*self.u[1:-1]*self.d_dx(self.A)/self.A[1:-1]
        
        if (self.diffusion.name != 'None'):
            self.diffusion.step(self, dt/n_s)
        
        """
        for i in range(0,3):
            self.q[i][1:-1] -= dt/n_s*self.q[i][1:-1]*self.d_dx(self.w)
        
        for i in range(0,self.ndim):
            self.X[i] += dt/n_s*self.w*self.ex[i]
            self.update_x()
            self.update_A()
        """                   
        
        for source in self.sources:
            source.step(self, dt/n_s)
        
        
        self.time += dt/n_s
        self.set_boundary()
        
        if (n_s > 1):
            for i in range(0,n_s-1):
                self.step(dt/n_s)


    def run(self,dt,tau,data=Data(),wait=True,verbose=True):
        if (type(data) == Data):
            data.set_scale(self.scale)
        
        mainloop = Mainloop(self,dt/self.scale.t,tau/self.scale.t,data,verbose)
        mainloop.start()
        if wait:
            mainloop.wait()
        return data
    
    
    def __del__(self):
        return
    
    

        
        

        
        
        