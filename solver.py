import pyopencl as cl
import numpy as np
import configure as cfg

class Solver():
    def __init__(self, A, b, method = "lusol", threshold = 1.):
        self.shape = A.shape
        self.block_size = 8;
        self.block_shape = (self.block_size, self.block_size, self.block_size)
        self.clinit()
        self.loadData(A, b)
        self.program = self.loadProgram(cfg.path+method+".cl")
        self.threshold = np.float32(threshold)

    def clinit(self):
        self.ctx = cl.create_some_context()       
        self.queue = cl.CommandQueue(self.ctx) 

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        kernel_params = {"nx": self.shape[2], "ny": self.shape[3], "nz": self.shape[4], "hx": self.shape[0], "hy": self.shape[1],
                        "block_size": self.block_size}
        return cl.Program(self.ctx, fstr % kernel_params).build()  
        
    def loadData(self, A, b):
        mf = cl.mem_flags        
        self.A = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=A)
        self.b = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)
        self.x = cl.Buffer(self.ctx, mf.READ_WRITE, b.nbytes)
        self.queue.finish()
   
    def run(self):        
        self.program.Solve(self.queue, self.shape[2:5], None, self.A, self.b, self.x, self.threshold)
        cl.enqueue_barrier(self.queue)
        return self
            
    def get(self):
        self.out = np.zeros((self.shape[0],)+self.shape[2:5], dtype = np.float32)
        cl.enqueue_read_buffer(self.queue, self.x, self.out)#.wait()
        self.queue.finish()
        return self.out

     

