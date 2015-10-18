#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>

__global__ void divideProcessKernel(float *d_A, int wA,int k)
{
	int iIdx = blockDim.x*blockIdx.x + threadIdx.x+k+1;
	int kIdx = k;;
	if (iIdx>wA-1)
	{
		return;
	}
	d_A[iIdx*wA+kIdx] = d_A[iIdx*wA+kIdx]/ d_A[kIdx*wA+kIdx];	
}

// Kernel by using shared memory
__global__ void updateProcessKernel_shared(float *d_A, int wA,int k)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x+k+1;
	int j = blockDim.y*blockIdx.y + threadIdx.y+k+1;

	if (i>wA-1 || j>wA-1)
	{
		return;
	}

	extern __shared__ float shared[];
	if (threadIdx.x==0)
	{
		shared[threadIdx.y] = d_A[j*wA+k];
		shared[blockDim.x+threadIdx.y] = d_A[k*wA+j];
	}
	__syncthreads();

	int idx = i*wA+j;
	d_A[idx] = d_A[idx] - shared[threadIdx.x]*shared[threadIdx.y+blockDim.x];
}

// GPU version2
void LUDecomposition_GPU_shared(float *d_A,int wA)
{
	dim3 ThreadDiv(512,1,1);
	dim3 BlockDiv((wA+ThreadDiv.x-1)/ThreadDiv.x,1,1);
	for (int k=0; k<wA; k++ )
	{
		divideProcessKernel<<<BlockDiv,ThreadDiv>>>(d_A,wA,k);

		dim3 ThreadUpdate(32,16,1);
		dim3 BlockUpdate((wA+ThreadUpdate.x-k-1)/ThreadUpdate.x,(wA+ThreadUpdate.x-k-1)/ThreadUpdate.y,1);
		updateProcessKernel_shared<<<BlockUpdate,ThreadUpdate,(ThreadUpdate.x + ThreadUpdate.y) * sizeof(float)>>>(d_A,wA,k);
	}
}

//vertify result
void VertifyResult(float *LURes, float *A,int wA)
{
	float *MulRes = new float[wA*wA];
	memset(MulRes,0,sizeof(float)*wA*wA);
	float temp;
	for (int i=0; i<wA; i++)//––
	{
		for (int j=0; j<wA; j++)//¡–
		{
			for (int ii=0; ii<=i; ii++)
			{
				if (i==ii)
				{
					temp = 1;
				}
				else
					temp = LURes[i*wA+ii];
				if (ii>j)
				{
					continue;
				}
				MulRes[i*wA+j] += temp*LURes[ii*wA+j];
			}			
		}
	}
	float temp2;
	bool bError = false;
	for (int i=0; i<wA; i++)//––
	{
		for (int j=0; j<wA; j++)//¡–
		{
			temp2 = abs(MulRes[i*wA+j] - A[i*wA+j]);
			if (temp2 > 1.000000E-01)
			{
				printf("Error:%f,%d %d,\n",temp2,i,j);
				bError = true;
			}

		}
	}

	if (!bError)
	{
		printf("Pass!\n");
	}
}

void GenSimData(int wA)
{
	float *A = new float[wA*wA];
	srand(time(NULL));
	for (int i=0; i<wA; i++)
	{
		for (int j=0; j<wA; j++)
		{
			A[i*wA+j]  = j;//rand()%99;
			if (A[i*wA+j] ==0)
			{
				A[i*wA+j] ++;
			}
		}
	}
	// Save Test Date
	FILE *fp;
	fp = fopen("Input.txt","w");
	if (fp == NULL)
	{
		return;
	}
	for (int i=0; i<wA; i++)
	{
		for (int j=0; j<wA; j++)
		{
			fprintf(fp,"%f ",A[i*wA+j]);
		}
		fprintf(fp,"\n");

	}
	fclose(fp);
	delete[] A;
	A = NULL;
}

bool ReadSimData(float *A, int wA)
{
	// Read Test Date
	FILE *fp;
	fp = fopen("Input.txt","r");
	if (fp == NULL)
	{
		return false;
	}
	for (int i=0; i<wA; i++)
	{
		for (int j=0; j<wA; j++)
		{
			fscanf(fp,"%f ",&A[i*wA+j]);
		}
	}
	fclose(fp);
	return true;
}

bool SaveLuResult(float *A, int wA)
{
	// Read Test Date
	FILE *fp;
	fp = fopen("LURes.txt","w");
	if (fp == NULL)
	{
		return false;
	}
	for (int i=0; i<wA; i++)
	{
		for (int j=0; j<wA; j++)
		{
			fprintf(fp,"%f ",A[i*wA+j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	return true;
}


int main()
{		
	// GPU
	int wA = 512;
	float *A = new float[wA*wA];
	GenSimData(wA); // Generate simulation Data and save file "Input.txt"
	ReadSimData(A,wA);
	float *LURes = new float[wA*wA];	
	float *d_A;
	cudaMalloc((void**)&d_A,sizeof(float)*wA*wA);
	cudaMemcpy(d_A,A,sizeof(float)*wA*wA,cudaMemcpyHostToDevice);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	LUDecomposition_GPU_shared(d_A,wA);

	cudaEventRecord(stop,0);
	cudaEventSynchronize( stop );
	float costtime;
	cudaEventElapsedTime(&costtime,start,stop);
	printf("Elapsed Time:%f\n",costtime);
	cudaMemcpy(LURes,d_A,sizeof(float)*wA*wA,cudaMemcpyDeviceToHost);

	SaveLuResult(LURes, wA);

	////vertify result
	VertifyResult(LURes,A,wA);

	return 0;
}