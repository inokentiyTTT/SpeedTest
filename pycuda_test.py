import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from time import time

n = 512
blockdim = 16, 16
griddim = int(n/blockdim[0]), int(n/blockdim[1])

L = 1.
h = L/n
dt = 0.1*h**2
nstp = 5000

mod = SourceModule("""
    __global__ void NextStpGPU(int* dN, double* dth2, double *u0, double *u)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        double uim1, uip1, ujm1, ujp1, u00, d2x, d2y;
        uim1 = exp(-10.0);
        if (i > 0)
    	       uim1 = u0[(i - 1) + dN[0]*j];
        else
    	       uim1 = 0.0;

        if (i < dN[0] - 1)
    	       uip1 = u0[(i + 1) + dN[0]*j];
        else
    	       uip1 = 0.0;

        if (j > 0)
    	       ujm1 = u0[i + dN[0]*(j - 1)];
        else
    	       ujm1 = 0.0;

        if (j < dN[0] - 1)
    	       ujp1 = u0[i + dN[0]*(j + 1)];
        else
    	       ujp1 = 1.0;

        u00 = u0[i + dN[0]*j];
        d2x = (uim1 - 2.0*u00 + uip1);
        d2y = (ujm1 - 2.0*u00 + ujp1);
        u[i + dN[0]*j] = u00 + dth2[0]*(d2x + d2y);
    }
    """)

u0 = np.full(n*n, 0., dtype = np.float64)
u = np.full(n*n, 0., dtype = np.float64)
nn = np.full(1, n, dtype = np.int64)
th2 = np.full(1, dt/h/h, dtype = np.float64)

st = time()

u0_gpu = cuda.to_device(u0)
u_gpu = cuda.to_device(u)
n_gpu = cuda.to_device(nn)
th2_gpu = cuda.to_device(th2)

func = mod.get_function("NextStpGPU")
for i in range(0, int(nstp/2)):
    func(n_gpu, th2_gpu, u0_gpu, u_gpu, 
         block=(blockdim[0],blockdim[1],1),grid=(griddim[0],griddim[1],1))
    func(n_gpu, th2_gpu, u_gpu, u0_gpu, 
         block=(blockdim[0],blockdim[1],1),grid=(griddim[0],griddim[1],1))

u0 = cuda.from_device_like(u0_gpu, u0)

print ('time on GPU = ', time() - st)
