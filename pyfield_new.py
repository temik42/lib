import numpy as np


class fLine:
    def __init__(self, field, verts, ends, color=(0,0,0), type='standard'):
        self.field = field
        self.verts = verts
        self.ends = ends
        self.color = color
        self.type = type
        
        if self.type == 'fan':
            self.linestyle = '--'
        else:
            self.linestyle = '-'
            

class fCharge:
    def __init__(self, field, q, x):
        self.field = field
        self.q = q
        self.x = x
        
        if self.q > 0:
            self.marker = '+'
            self.markersize = 10
            self.sign = 'pos'
            self.name = 'P'+str(field.n_pos+1)
            self.field.n_pos += 1
        else:
            self.marker = 'x'
            self.markersize = 8
            self.sign = 'neg'
            self.name = 'N'+str(field.n_neg+1)
            self.field.n_neg += 1

            
            
class fInfinity:
    def __init__(self, field, q):
        self.field = field
        self.q = q
        self.x = np.inf

        
        
class Field:
    def __init__(self, Q, X):
        self.set_charges(Q,X)

    def set_charges(self, q, x):
        self.n_dim = x.shape[1]
        self.n_charges = q.shape[0]
        self.n_pos = 0
        self.n_neg = 0
        
        self.Charges = []
        for i in range(0,self.n_charges):
            self.Charges += [fCharge(self, q[i], x[i])]
            
        self.Infinity = fInfinity(self, -np.sum(q))
        return self
        
    def X(self, sign = 'all'):
        return np.array([charge.x for charge in self.Charges if charge.sign == sign or sign == 'all'])
    
    def Q(self, sign = 'all'):
        return np.array([charge.q for charge in self.Charges if charge.sign == sign or sign == 'all'])
    
    def get(self, x, jac=False, minr=False):
        dx = (x - self.X())
        dr = np.sqrt(np.sum(dx**2, axis = 1)).clip(min = 1e-10)
        f = self.Q()/dr**3
        out = (np.dot(f, dx),)
        if jac:
            out = out + (np.sum(f)*np.identity(self.n_dim)-3*np.dot(f/dr**2*dx.T, dx),)
        if minr:
            out = out + (np.min(dr),np.argmin(dr),)
        return out
    

    def rk2(self, x, maxi = 200, step = 0.5):
        stack = x
        i = 0      
        while i < maxi:
            p, jac, minr, argminr = self.get(x, True, True)
            dx = p/np.linalg.norm(p)*step
            
            if minr < abs(step):
                stack = np.column_stack((stack,self.X()[argminr]))
                return (stack, self.Charges[argminr])
            else:                    
                p += 0.5*np.dot(jac, dx)
                dx = p/np.linalg.norm(p)*step
                x += dx
                stack = np.column_stack((stack,x))
            i += 1
        return (stack, self.Infinity)

    
    def fline(self, x, maxi = 200, step = 0.5):
        f1 = self.rk2(x, maxi = maxi, step = step)
        f2 = self.rk2(x, maxi = maxi, step = -step)
        if f1[1] != f2[1]:
            verts = np.column_stack((np.fliplr(f1[0])[:,:-1], f2[0]))
            ends = [f1[1],f2[1]]
            return fLine(self, verts, ends)




        