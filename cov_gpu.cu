#include "cov_gpu.hpp"
#include <stdio.h>

__device__ REAL cov_val_d(int n_dim, REAL *x, REAL *y, REAL *hypers)
{
	REAL scale = hypers[n_dim];
	REAL s = 0.0;
	for (unsigned i=0; i<n_dim; i++)
	{
		REAL d = (x[i]-y[i])/hypers[i];
		s += d * d;
	}
	return scale * exp(-0.5*s);
}

__global__ void cov_val_d_wrapper(REAL *result_d, int n_dim, REAL *x, REAL *y, REAL *hypers)
{
	*result_d = cov_val_d(n_dim,x,y,hypers);
}

void cov_val_wrapper(REAL *result_d, int n_dim, REAL *x, REAL *y, REAL *hypers)
{
	cov_val_d_wrapper<<<1,1>>>(result_d, n_dim, x, y, hypers);
}

// Computes the vector of covariances with a new point with (the vector 'k' in the notation I have been using)
// could use thrust device vectors or similar
// just values for now -- fix when working
__global__ void cov_all_kernel(REAL *result, int N, int n_dim, REAL *xnew, REAL *xs, REAL *theta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<N) {
		result[i] = cov_val_d(n_dim, xnew, xs + n_dim*i, theta);
	}
}

__global__ void cov_batch(REAL *result, int Nnew, int N, int n_dim, REAL *xsnew, 
			  REAL *xs, REAL *theta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i<N && j<Nnew) {
		result[j+Nnew*i] = cov_val_d(n_dim, xsnew + n_dim*j, xs + n_dim*i, theta);
	}
}

// wrapper
void cov_all_wrapper(REAL *result, int N, int n_dim, REAL *xnew, REAL *xs, REAL *theta)
{
	const int threads_per_block = 256;
	cov_all_kernel <<< 10, threads_per_block >>> (result, N, n_dim, xnew, xs, theta);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

void cov_batch_wrapper(REAL *result, int Nnew, int N, int n_dim, REAL *xsnew, 
			  REAL *xs, REAL *theta)
{
	dim3 threads_per_block(8,32);
	dim3 blocks(250,625);
	cov_batch <<< blocks, threads_per_block >>> (result, Nnew, N, n_dim, xsnew, xs, theta);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
