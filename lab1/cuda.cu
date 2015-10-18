 
#include <stdio.h>
#include <cuda.h>

// Kernel that executes on the CUDA device
__global__ void temp_calc(float *a, float *b, int N, int edge) // a: Source array, b: Target array, N: Total size, edge: Length of edge
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Calculates the row and column number
  int row = idx / (edge+1);
  int col = idx - row * (edge+1);
  if (idx<N) {
    if(row>0 && row<edge && col>0 && col<edge) // Not on the edges
      b[row*(edge+1)+col]=(a[(row-1)*(edge+1)+col]+a[(row+1)*(edge+1)+col]+a[row*(edge+1)+col-1]+a[row*(edge+1)+col+1])/4.0;
  }
}


// main routine that executes on the host
int main(void)
{
  //clock_t start = clock();
  float *a_h, *a_d, *b_d;  // Pointer to host & device arrays
  int edge = 1000; // Can be changed
  const int N = (edge+1) * (edge+1);  // Number of elements in arrays
  size_t size = N * sizeof(float);
  a_h = (float *)malloc(size);        // Allocate array on host
  cudaMalloc((void **) &a_d, size);   // Allocate array a on device
  cudaMalloc((void **) &b_d, size);   // Allocate array b on device

  // Initialize host array
  for (int i=0; i<=edge; i++) {
    for (int j=0; j<=edge; j++){
      if(i==0){
        if(j>=10 && j<=30){
          a_h[i*(edge+1)+j]=150.0;
        }
        else{
          a_h[i*(edge+1)+j]=80.0;
        }
      }
      else{
        if(i==edge || j==0 || j==edge){
          a_h[i*(edge+1)+j]=80.0;
        }
        else{
          a_h[i*(edge+1)+j]=0.0;
        }
      }
    }
  }

  // Initialize block size and block number
  int block_size = 256;
  int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
  
  // Iteration
  int iter = 500; // Can be changed
  for (int i=0;i<iter;i++){
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); // Copy the host array to the CUDA
    cudaMemcpy(b_d, a_h, size, cudaMemcpyHostToDevice);
    temp_calc <<< n_blocks, block_size >>> (a_d, b_d, N, edge); // Calculate the values on CUDA
    cudaMemcpy(a_h, b_d, sizeof(float)*N, cudaMemcpyDeviceToHost); // Retrieve result from device and store it in host array
  }
/*
  // Print results
  for (int i=0; i<=edge; i++) {
    for (int j=0; j<=edge; j++)
      printf("%f ", a_h[i*(edge+1)+j]);
    printf("\n");
  }

  clock_t end = (clock() - start)/1000;
  printf("time: %ldms\n", end);
*/
  // Cleanup
  free(a_h); cudaFree(a_d);
}