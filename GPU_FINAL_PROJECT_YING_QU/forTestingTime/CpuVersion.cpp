#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// CPU version
int LUDecomposition(float *A, const int size)
{
	int i,j,k;
	for (k = 0; k < size; k++)
	{
		for (i = k+1; i < size; i++)
		{
			A[i*size+k] = A[i*size+k] / A[k*size+k];
		}

		for (i = k+1; i < size; i++)
			for (j = k+1; j < size; j++)
				A[i*size+j] = A[i*size+j] - A[i*size+k] * A[k*size+j];
	}
	return 1;
} 

//vertify result
void VertifyResult(float *LURes, float *A,int wA)
{
	float *MulRes = new float[wA*wA];
	memset(MulRes,0,sizeof(float)*wA*wA);
	float temp;
	for (int i=0; i<wA; i++)//行
	{
		for (int j=0; j<wA; j++)//列
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
	for (int i=0; i<wA; i++)//行
	{
		for (int j=0; j<wA; j++)//列
		{
			temp2 = abs(MulRes[i*wA+j] - A[i*wA+j]);
			if (temp2 >1.000000E-01)
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
	float *LURes = new float[wA*wA];
	//GenSimData(wA); // Generate simulation Data and save file "Input.txt"
	//ReadSimData(A,wA);
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

	clock_t start=clock();

	memcpy(LURes,A,sizeof(float)*wA*wA);
	LUDecomposition(LURes, wA);

	clock_t end=clock();
	
	printf("Elapsed Time:%dms\n",(end-start));

	////vertify result
	//VertifyResult(LURes,A,wA);

	//system("PAUSE");	
    return 0;
}