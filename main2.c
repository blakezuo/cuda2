#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cassert>

#define ITERATIONS 10
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1
#define WARP_PER_BLOCK 8
#define WARP_SIZE 32

using namespace std;

void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else 
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else 
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    if(i<10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>0.0005)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}
//The CSR-format matrix is dimXdim that has n non-zero elements.
void initMatrix(int *row, int *col, float *data, int n, int dim){
      int nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    for (int i = 0; i < dim; i++)
    {
        row[i] = nnzAssigned;
        for (int j = 0; j < dim; j++)
        {
            int numEntriesLeft = (dim * dim) - ((i * dim) + j);
            int needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                col[nnzAssigned] = j;
    data[nnzAssigned] = 1;
                nnzAssigned++;
            }
        }
   }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    row[dim] = n;
    assert(nnzAssigned == n);
}

__global__ void spmv(int* row, int* col, float* data, float* vec, float* res, int dim, int n){
  int i = blockIdx.x * WARP_PER_BLOCK + threadIdx.x / WARP_SIZE;
  int warp = threadIdx.x % WARP_SIZE;
  if(i<dim){
    __shared__ float sum[6][WARP_SIZE];
    for(int j=0;j<6; j++) sum[j][warp] = 0.0;
    float tmp = 0;
    for(int j=row[i] + warp; j<row[i+1];j=j+WARP_SIZE)
    {
        int colTmp = col[j];
        tmp += data[j] * vec[colTmp];
    }
    printf("%f\n",tmp);
    sum[0][warp] = tmp;
    __syncthreads();
    int times = 1,l = WARP_SIZE / 2;
    while(warp / l == 0)
    {
        times ++;
        l /= 2;
    }
    int scale = WARP_SIZE / 2;
    for(int j=1;j<=times;j ++)
    {
        sum[j][warp - scale] += sum[j-1][warp];
        scale /= 2;
    }
    __syncthreads();
    res[i] = sum[5][0];
  }
}





int main(){

  int dim=20000;
  int n=dim*dim/100;
  int *row = (int*)malloc(sizeof(int)*(dim+1));
  int *col = (int*)malloc(sizeof(int)*n);
  float *data = (float*)malloc(sizeof(float)*n);
  initMatrix(row, col, data, n, dim);

  float *vec = (float*)malloc(sizeof(float)*dim);
  for(int i=0; i<dim; i++){
    vec[i]=1;
  }

  float *result = (float*)malloc(sizeof(float)*dim);
  float *result_gpu_res = (float*)malloc(sizeof(float)*dim);

  for(int i=0; i<dim; i++){
    float t = 0;
    for(int j=row[i]; j<row[i+1]; j++){
      int colNum = col[j];
      t += data[j] * vec[colNum];
    }
    result[i] = t;
  }

  int *row_gpu;
  int *col_gpu;
  float *data_gpu;
  float *vec_gpu;
  float *result_gpu;
  cudaMalloc( (void **)&row_gpu, sizeof(int)*(dim+1));
  cudaMalloc( (void **)&col_gpu, sizeof(int)*n);
  cudaMalloc( (void **)&data_gpu, sizeof(float)*n);
  cudaMalloc( (void **)&vec_gpu, sizeof(float)*dim);
  cudaMalloc( (void **)&result_gpu, sizeof(float)*dim);
  cudaMemcpy(row_gpu, row, sizeof(int)*(dim+1), cudaMemcpyHostToDevice);
  cudaMemcpy(col_gpu, col, sizeof(int)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(data_gpu, data, sizeof(float)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(vec_gpu, vec, sizeof(float)*dim, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil(((float)(dim*WARP_SIZE)/ (float)DIM_THREAD_BLOCK_X)), 1);

  spmv<<<grid,block>>>(row_gpu, col_gpu, data_gpu, vec_gpu, result_gpu, dim, n);
  cudaThreadSynchronize();
  cudaMemcpy(result_gpu_res, result_gpu, sizeof(float)*dim, cudaMemcpyDeviceToHost);
  compare(result, result_gpu_res, dim);



  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){

    spmv<<<grid,block>>>(row_gpu, col_gpu, data_gpu, vec_gpu, result_gpu, dim, n);

  }
  cudaThreadSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double flops = 2 * (double)n;
  double gflopsPerSecond = flops/(1000000000)/time;
  double dataCopy = sizeof(int)*dim + sizeof(int)*n + sizeof(float)*n + sizeof(float)*dim*2;
  double bandwidth = dataCopy/time/1000000000;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",bandwidth );
  printf("GB=%lf\n",dataCopy/1000000000);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("time(s)=%lf\n",time);
  return 0;
}
