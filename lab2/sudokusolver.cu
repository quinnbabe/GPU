#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> 
#include <cuda.h>
#pragma comment(lib, "cudart.lib")

void ReadInputData(char *fileName, int *matrix)
{
	FILE *fp;
	char temp[10];
	fp = fopen(fileName, "r");
	for (int i=0; i<9; i++)
	{
		fscanf(fp,"%s",temp);
		for (int j=0; j<9; j++)
		{
			matrix[i*9+j] = temp[j]-'0';
		}
	}	
	fclose(fp);
}
void OutputResultData(char *fileName, int *d_matrix)
{
	int matrix[81] = {0};
	cudaMemcpy(matrix, d_matrix, 81* sizeof(int), cudaMemcpyDeviceToHost);

	FILE *fp;
	//char temp[10];
	fp = fopen(fileName, "w");
	for (int i=0; i<9; i++)
	{	
		for (int j=0; j<9; j++)
		{
			fprintf(fp,"%d",matrix[i*9+j]);			
		}
		fprintf(fp,"\n");
	}	
	fprintf(fp,"\n");
	fclose(fp);
}
//Outputing the soduku matrix according to appropriate format
void testRes(int *d_matrix)
{
	int matrix[81] = {0};
	cudaMemcpy(matrix, d_matrix, 81* sizeof(int), cudaMemcpyDeviceToHost);

	int i,j;
	for(i=0;i<9;i++)
	{
		for(j=0;j<9;j++)
		{
			printf("%d ",matrix[i*9+j]);
		}
		printf("\n");			
	}
	printf("\n");
}
__device__ void CalSudokuCondense(int i, int j, int *m, int n)
{
	int kk;
	int breakTrue = false;
	// determine row
	for (int ii=0; ii<9; ii++)
	{
		for (kk=1; kk<10; kk++)
		{
			if (m[ii*90+j*10+kk] == n)
			{
				m[ii*90+j*10+kk] = 0;
				atomicSub(&m[ii*90+j*10],1);
				breakTrue = true;
				break;
			}			
		}
		//if (breakTrue)
		//{
		//	for (int tt = kk; tt<8; tt++)
		//	{
		//		m[ii*90+j*9+tt] = m[ii*90+j*9+tt+1];
		//	}
		//}
	}
	//syncthreads();
	//determine column
	for (int jj=0; jj<9; jj++)
	{
		for (kk=1; kk<10; kk++)
		{
			if (m[i*90+jj*10+kk] == n)
			{
				m[i*90+jj*10+kk] = 0;
				//m[i*90+jj*10] = m[i*90+jj*10]-1;
				atomicSub(&m[i*90+jj*10],1);
				breakTrue = true;
				break;
			}			
		}
		//if (breakTrue)
		//{
		//	for (int tt = kk; tt<8; tt++)
		//	{
		//		m[i*90+jj*9+tt] = m[i*90+jj*9+tt+1];
		//	}
		//}
	}
	//syncthreads();
	// determine 3*3 grid
	int idxx = i/3*3;
	int idxy = j/3*3;
	for (int ii=idxx; ii<idxx+3; ii++)
	{
		for (int jj=idxy; jj<idxy+3; jj++)
		{
			for (kk=1; kk<10; kk++)
			{
				if (m[ii*90+jj*10+kk] == n)
				{
					m[ii*90+jj*10+kk] = 0;
					//m[ii*90+jj*10] = m[ii*90+jj*10]-1;
					atomicSub(&m[ii*90+jj*10],1);
					breakTrue = true;
					break;
				}			
			}
			//if (breakTrue)
			//{
			//	for (int tt = kk; tt<8; tt++)
			//	{
			//		m[ii*90+jj*9+tt] = m[ii*90+jj*9+tt+1];
			//	}
			//}
		}
	}
	//syncthreads();
}

__device__ void CalSudoku(int i, int j, int *sudoku_matrix, int mArray[10])
{
	int availableTemp[9]={1,2,3,4,5,6,7,8,9};
	if(0==sudoku_matrix[i*9+j])
	{
		//the empty grids to fill
		//Finding out the eligible numbers, and locating them into array available[]
		//row(i,matrix,available);
		for (int k=0; k<9; k++)
		{
			if (sudoku_matrix[i*9+k] !=0)
			{
				availableTemp[sudoku_matrix[i*9+k]-1] = 0;
			}			
		}
		//column(j,matrix,available);
		for (int k=0; k<9; k++)
		{
			if (sudoku_matrix[k*9+j] !=0)
			{
				availableTemp[sudoku_matrix[k*9+j]-1] = 0;
			}			
		}
		//block(i,j,matrix,available);
		// determine 3*3 grid
		int idxx = i/3*3;
		int idxy = j/3*3;
		for (int ii=idxx; ii<idxx+3; ii++)
		{
			for (int jj=idxy; jj<idxy+3; jj++)
			{
				if (sudoku_matrix[ii*9+jj] !=0)
				{
					availableTemp[sudoku_matrix[ii*9+jj]-1] = 0;
				}	
			}
		}
		//condense(available);
		int tt=1;
		for (int k=0; k<9; k++)
		{
			if (availableTemp[k] != 0)
			{
				mArray[tt]= availableTemp[k];
				tt++;
			}
		}
		mArray[0]=tt-1;
	}
}
__global__ void CheckEndCUDA(int *sudokuMatrix, int *Res)
{	
	int i = blockIdx.x;
	//int i = threadIdx.y;
	int j = threadIdx.x;
	Res[0] = 1;
	//int sum=0;
	if (sudokuMatrix[i*9+j] == 0)
	{
		Res[0] = 0;
	}	
	return;
}
__global__ void SudokuCUDA(int *sudokuMatrix, int *m)
{	
	int i = blockIdx.x;
	//int i = threadIdx.y;
	int j = threadIdx.x;	
	CalSudoku(i, j, sudokuMatrix, m+(i*90+j*10));
}
__global__ void SudokuCondenseCUDA(int *sudokuMatrix, int *m)
{	
	int i = blockIdx.x;
	//int i = threadIdx.y;
	int j = threadIdx.x;	
	//if (m[i*90+j*9] != 0 && m[i*90+j*9+1] == 0)
	//{
	//	sudokuMatrix[i*9+j] = m[i*90+j*9];
	//	 m[i*90+j*9] = 0;
	//	 //printf("%d",sudokuMatrix[i*9+j])
	//	 CalSudokuCondense(i, j, m,sudokuMatrix[i*9+j]);
	//}
	//int count=0;
	//int dataTemp = 0;
	//for (int k=1; k<10;k++)
	//{
	//	if (m[i*90+j*10+k] != 0)
	//	{
	//		count++;
	//		dataTemp = m[i*90+j*10+k];
	//	}
	//}
	if (m[i*90+j*10] == 1)
	//if(count == 1)
	{
		sudokuMatrix[i*9+j] = m[i*90+j*10+1];
		//CalSudokuCondense(i, j, m,sudokuMatrix[i*9+j]);
	}	
}

void TestM(char *fileName, int *m)
{
	FILE *fp;
	//char temp[10];
	fp = fopen(fileName, "w");
	for (int iii=0; iii<9;iii++)
	{
		for (int jjj=0; jjj<9; jjj++)
		{
			for (int kkk=0; kkk<10; kkk++)
			{
				fprintf(fp, "%d ", m[iii*90+jjj*10+kkk]);
			}
			fprintf(fp,"\n");
		}
	}
	fclose(fp);
}

int main ()
{
	//the matrix for sudoku
	//Items 0 represent the empty grids
	int sudoku_matrix[81]={0};
	ReadInputData("sudokusolver.txt", sudoku_matrix); // read input data

	int *dev_sudoku_matrix;		
	int memSizeMatirx = sizeof(int)*81;
	cudaMalloc((void**)&dev_sudoku_matrix, memSizeMatirx); // GPU buffer store computing array

	int *dev_m;
	int memSizeM = sizeof(int)*9*9*10;
	cudaMalloc((void**)&dev_m, memSizeM);// GPU buffer store the process computing array

	int ResEnd = 0; // determine whether the computing is completed
	int *dev_ResEnd;
	cudaMalloc((void**)&dev_ResEnd, sizeof(int));
	cudaMemcpy(dev_ResEnd,&ResEnd,sizeof(int),cudaMemcpyHostToDevice);

	cudaMemcpy(dev_sudoku_matrix,sudoku_matrix,memSizeMatirx,cudaMemcpyHostToDevice); // copy data to GPU
	cudaMemset(dev_m,0,memSizeM);	//clear processing array
	
	//computing the runtime
	//float elapsedTimeInMs = 0.0f;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	int threadDim = 9;
	int blockDim = 9;
	
	//int matrix[90*9] = {0};


	while(!ResEnd)
	{

		SudokuCUDA<<<blockDim, threadDim>>>(dev_sudoku_matrix, dev_m); // computing all possible answers

		//cudaMemcpy(matrix, dev_m, memSizeM, cudaMemcpyDeviceToHost);
		//TestM("M1.txt", matrix);	

		SudokuCondenseCUDA<<<blockDim, threadDim>>>(dev_sudoku_matrix, dev_m); // simplify the function according to the only computed value



		CheckEndCUDA<<<blockDim, threadDim>>>(dev_sudoku_matrix, dev_ResEnd);  // check whether it is completed	
		cudaMemcpy(&ResEnd, dev_ResEnd, sizeof(int), cudaMemcpyDeviceToHost);		
	}

	//computing the runtime
	//cudaEventRecord(stop, 0);
	//cudaDeviceSynchronize();
	//cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	//printf("GPU used %fms\n", elapsedTimeInMs);

	OutputResultData("sudokusolver.sol", dev_sudoku_matrix);
	//testRes(dev_sudoku_matrix);	
	cudaFree(dev_sudoku_matrix);
}

