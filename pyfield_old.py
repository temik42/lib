import numpy as np
from potential import PField

def hessian(X):
    J = np.gradient(X, edge_order = 2)
        
    H = [[(-2*X + np.roll(X,-1,k)+np.roll(X,1,k)) if k == j else 
           (np.roll(J[j],-1,k)-np.roll(J[j],1,k))*0.5 for k in range(0,3)] for j in range(0,3)]
    
    H[0][0][0,:,:] = (2*X[0,:,:]-5*X[1,:,:]+4*X[2,:,:]-X[3,:,:])
    H[0][0][-1,:,:] = (2*X[-1,:,:]-5*X[-2,:,:]+4*X[-3,:,:]-X[-4,:,:])
    H[1][1][:,0,:] = (2*X[:,0,:]-5*X[:,1,:]+4*X[:,2,:]-X[:,3,:])
    H[1][1][:,-1,:] = (2*X[:,-1,:]-5*X[:,-2,:]+4*X[:,-3,:]-X[:,-4,:])
    H[2][2][:,:,0] = (2*X[:,:,0]-5*X[:,:,1]+4*X[:,:,2]-X[:,:,3])
    H[2][2][:,:,-1] = (2*X[:,:,-1]-5*X[:,:,-2]+4*X[:,:,-3]-X[:,:,-4])
        
    for j in (1,2):
        H[j][0][0,:,:] = (-1.5*J[j][0,:,:]+2*J[j][1,:,:]-0.5*J[j][2,:,:])
        H[j][0][-1,:,:] = (1.5*J[j][-1,:,:]-2*J[j][-2,:,:]+0.5*J[j][-3,:,:])
        H[0][j][0,:,:] = H[j][0][0,:,:]
        H[0][j][-1,:,:] = H[j][0][-1,:,:]
        
    H[2][1][:,0,:] = (-1.5*J[2][:,0,:]+2*J[2][:,1,:]-0.5*J[2][:,2,:])
    H[2][1][:,-1,:] = (1.5*J[2][:,-1,:]-2*J[2][:,-2,:]+0.5*J[2][:,-3,:])
    H[1][2][:,0,:] = H[2][1][:,0,:]
    H[1][2][:,-1,:] = H[2][1][:,-1,:]
    
    return (H, J)





class Field:
    def __init__(self, data, nz = 64):
        phi = PField(data, nz)    
        hes = hessian(-phi)
        self.B = np.array(hes[1])
        self.dB = np.array(hes[0])
        self.dim = phi.shape
    
    
    def update(self, x):  
        self._x = x
        xn = np.round(self._x).astype(np.int32)
        dxn = self._x - xn       
        self._jacobian = self.dB[:,:,xn[0],xn[1],xn[2]]
        self._p = self.B[:,xn[0],xn[1],xn[2]] + np.dot(self._jacobian, dxn)
        self._minr = np.min([np.abs(self._x),np.abs(np.array(self.dim)-self._x-1)])
    
    
    def Update(self, X):  
        self._X = X
        Xn = np.round(self._X).astype(np.int32)
        dXn = self._X - Xn       
        self._Jacobian = self.dB[:,:,Xn[0,:],Xn[1,:],Xn[2,:]]
        self._P = self.B[:,Xn[0,:],Xn[1,:],Xn[2,:]] + np.sum(self._Jacobian*dXn, axis = 1)
        
        r = [self.dim[i]/2. - np.abs(self._X[i,:] - self.dim[i]/2.) for i in range(0,3)]
        self._minr = np.min(r, axis = 0)
        self._argminr = np.argmin(r, axis = 0)
    
    
    def rk2(self, x, mul = 1, maxi = 200, step = 0.5):
        self.update(x)
        stack = self._x
        
        i = 0      
        while i < maxi:       
            dx = self._p/linalg.norm(self._p)*step*mul
            
            if self._minr < step:
                return stack
            
            dp = np.dot(self._jacobian, dx)                 
            self._p = self._p + 0.5*dp
            dx = self._p/linalg.norm(self._p)*step*mul
            
            lp = linalg.norm(self._p)
            ldp = linalg.norm(dp)
            
            self.update(self._x + dx)
            if (self._minr < step) or (lp < ldp):
                return stack
                
            stack = np.column_stack((stack,self._x))
            i += 1
        return stack
    
    
    def Rk2(self, X, mul = 1, maxi = 200, step = 0.5):
        self.Update(X)
        NX = X.shape[1]
        
        absP = np.sqrt(np.sum(self._P**2., axis=0))
        out = [(X[:,i],absP[i]) for i in range(0,NX)]
        #out1 = [np.sqrt(np.sum(self._P**2., axis=0))[i] for i in range(0,NX)]
        
        ids = np.where(np.ones(NX))[0]      
        
        i = 0
        while (i < maxi):                   
            absP = np.sqrt(np.sum(self._P**2., axis=0))
            dX = self._P/absP*step*mul
            dP = np.sum(self._Jacobian*dX, axis = 1)
            self._P = self._P + 0.5*dP
            absP = np.sqrt(np.sum(self._P**2., axis=0))
            dX = self._P/absP*step*mul
            dP = np.sum(self._Jacobian*dX, axis = 1)
            absdP = np.sqrt(np.sum(dP**2., axis=0))
            
            self.Update(self._X + dX)

            t = np.where(np.all([self._minr >= 2.*step, absP >= 0.5*absdP],0))[0]

            for ii in range(0,ids.shape[0]):
                if self._minr[ii] < 2.*step:
                    self._X[self._argminr[ii],ii] = 0 if self._X[self._argminr[ii],ii] < 2*step else self.dim[self._argminr[ii]]-1
                out[ids[ii]] = (np.column_stack((out[ids[ii]][0], self._X[:,ii])),np.append(out[ids[ii]][1],absP[ii]))
                #out1[ids[ii]] = np.column_stack((out1[ids[ii]], absP[ii]))
            
            self._P = self._P[:,t]
            self._X = self._X[:,t]  
            self._Jacobian = self._Jacobian[:,:,t]            
            
            ids = ids[t]
            i += 1
    
        return out
    
    
    def fline(self, x):
        f1 = self.rk2(x)
        f2 = self.rk2(x, mul = -1)
        
        nf1 = len(f1.shape)
        nf2 = len(f2.shape)
        
        if (nf1 != 1) and (nf2 != 1):
            return np.column_stack((np.fliplr(f1), f2[:,1:]))
        if (nf1 != 1):
            return f1
        if (nf2 != 1):
            return f2
        return x
    
    
    def Fline(self, X, step = 0.5):
        NX = X.shape[1]
        
        F1 = self.Rk2(X, step = step)
        F2 = self.Rk2(X, mul = -1, step = step)
        
        out = []
        
        
        for i in range(0,NX):
            NF1 = len(F1[i][0].shape)
            NF2 = len(F2[i][0].shape)
            
            if (NF1 != 1) and (NF2 != 1):
                #out = out + [np.column_stack((np.fliplr(F1[i]), F2[i][:,1:]))]
                out = out + [(np.column_stack((np.fliplr(F1[i][0]), F2[i][0][:,1:])),
                             np.concatenate((F1[i][1][::-1], F2[i][1][1:])))]
                
            if (NF1 == 1) and (NF2 != 1):
                out = out + [F2[i]]
                
            if (NF1 != 1) and (NF2 == 1):
                out = out + [F1[i]]

        return out
        
