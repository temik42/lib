import numpy as np

     
class Interface(object):
    def __init__(self, lam, S, d, alpha):
        self.alpha = alpha
            
    def connect(self, element1, element2):
        self.elements = (element1,element2)
        e = [self.elements[i].e for i in range(0,2)]
        self.e2 = e[1]/(e[0]+e[1]-e[0]*e[1])
        self.E = e[0]*self.e2
            
    def apply(self, A, b):  
        Lam = self.Lam()
        
        for i in range(0,2):
            if (self.elements[i].type == 'Node'):
                A[self.elements[i].idx,self.elements[i].idx] += Lam*(1+self.alpha)
                b[self.elements[i].idx] += Lam*self.alpha*self.elements[i].T             
                if (self.elements[i-1].type == 'Node'):
                    A[self.elements[i].idx,self.elements[i-1].idx] -= Lam*(1+self.alpha)
                    b[self.elements[i].idx] -= Lam*self.alpha*self.elements[i-1].T     
                else:
                    b[self.elements[i].idx] += Lam*self.elements[i-1].T    



class Illuminance(Interface):
    def __init__(self, P, S):
        Interface.__init__(self, P, S, 0, -1)
        
        def Lam():
            T = [self.elements[i].T for i in range(0,2)]
            return P*S/(T[0]-T[1])*self.e2
        self.Lam = Lam      
            
class Transfer(Interface):
    def __init__(self, P, S):
        Interface.__init__(self, P, S, 0, -1)
        
        def Lam():
            T = [self.elements[i].T for i in range(0,2)]
            return P*S/(T[0]-T[1])
        self.Lam = Lam

class Conduction(Interface):
    def __init__(self, lam, S, d=1):
        Interface.__init__(self, lam, S, d, 0)
        
        self.Lam = lambda: lam*S/d
        
class Radiation(Interface):
    def __init__(self, S):
        sigma = 5.67e-8
        Interface.__init__(self, sigma, S, 0, 10)
        
        def Lam():
            T = [self.elements[i].T for i in range(0,2)]
            return sigma*S*(T[0]**2+T[1]**2)*(T[0]+T[1])*self.E
        self.Lam = Lam
        
        
                    
class Element(object):
    def __init__(self, type, T, e, name = 'None'):
        self.neighbours = []
        self.type = type
        self.T = T
        self.e = e
        self.name = name
        
    def connect(self, element, interface):       
        interface.connect(self, element)
    
    def set_idx(self, idx):
        self.idx = idx
        return self
    
    
    
class Node(Element):
    def __init__(self, T = 300, e = 1, name = 'None'):
        Element.__init__(self, 'Node', T, e, name)
    

    
class Sink(Element):
    def __init__(self, T = 0, e = 1, name = 'None'):
        Element.__init__(self, 'Sink', T, e, name)    

      
    
        
        
    
def Calculate(elements, interfaces, niter = 100):
    n = 0
    for element in elements:
        if element.type == 'Node':
            element.set_idx(n)
            n += 1
            
    for i in range(0,niter):
        A = np.zeros((n,n), dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)
        
        for interface in interfaces:
            interface.apply(A,b)

        T = np.dot(np.linalg.inv(A),b)
        for j in range(0,n):
            for element in elements:
                if element.type == 'Node':
                    if element.idx == j:
                        element.T = T[j]
                        
    print A, b
    return T