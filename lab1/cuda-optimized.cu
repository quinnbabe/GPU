#include <stdio.h>
#include <cuda.h>
#include <math.h>

typedef unsigned long long ll;

void printSum(float * H, ll ss) {

	float all = 0;
	for (ll i = 0; i < ss; i++) {
		all += H[i];
		
	}
	printf("avg = %.3f\n", all/ss);
}
__device__ bool zeroArea (ll i, ll n, ll iter)
{
    ll row = i/(1+n);
    ll col = i%(1+n);
    return iter < row && row < n-iter && iter < col && col < n - iter;
}

__device__ ll mediumTrans(ll n, ll iter, ll i)
{
    ll temp = (n+2)*(n-iter-1)+1;
     if (i>= temp){ 
	return i-(n-2*iter-1)*(n-2*iter-1);
	}
    else if ( i<temp&&(n+1)*(iter+1)+iter<i) {
        return i-((i-(iter+1)*(n+2))/(n+1)+1)*(n-2*iter-1);
    }
    
    return i;
}

__device__ ll revertMediumTrans (ll n, ll iter, ll newIdx)
{
    ll temp = 3*iter*n+3*n-4*iter*iter-6*iter-2;
    if (newIdx >= temp){
        return newIdx + (n-2*iter-1)*(n-2*iter-1);
    }
    else if ( newIdx<temp &&(n+1)*(iter+1)+iter<newIdx) {
        return newIdx + ((newIdx-(n+2)*(iter+1))*(n-2*iter-1)/(2*(iter+1))+1);
    }
   
    return newIdx;
}

__device__ ll matrixTrans (ll n, ll idx)
{
    return n+2+idx+2*(idx/(n-1));
}


__device__ ll revertMatrixTrans (ll n, ll iter, ll midiumIdx)
{
    ll idx = midiumIdx;
    ll temp = 3*iter*n-4*iter*iter-4*iter;
	
	if(midiumIdx >= temp){
        idx = midiumIdx+(n-2*iter-1)*(n-2*iter-1);
        return matrixTrans(n, idx);
    }
	else if(midiumIdx< temp && iter*n-1<midiumIdx) {
        idx=((midiumIdx-iter*n)/(2*iter)+1)*(n-2*iter-1)+midiumIdx;
        return matrixTrans(n, idx);
    }
    
    return matrixTrans(n, idx);
}

ll transHost(ll n, ll iter, ll i)
{
    ll temp = (n+2)*(n-iter-1) + 1;
    if (i>= temp){ 
	return i-pow((n-2*iter-1),2);
	}
    else if ( i<temp && ((n+1)*(iter + 1)+iter) < i){ 
	    return i-(n-2*iter-1)*((i-(n+2)*(iter+1))/(n+1)+1);
	}
    return i;
}

void printDeviceInfo(ll index) {
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) {
		printf("Get device properties error!");
		exit(EXIT_FAILURE);
	}
	printf("\n#Device %d#\n\nName: %s\nWarp size: %d\nMax threads per block: %d\nMax thread dimension: x => %d, y => %d, z=> %d\n Max grid dimension: x => %d, y = %d, z => %d\n",index, prop.name, prop.warpSize, prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

}

__global__ void calculateKernel(float * A, float * B, ll qq) {

        int gridnum = blockIdx.x * blockDim.x + threadIdx.x;
        int indexindex =qq+gridnum%(qq-2)+qq*(gridnum/(qq-2))+1;

        if ( gridnum<(qq-2)*(qq-2))  {
                B[indexindex] = (A[indexindex-qq]+A[indexindex+qq]+ A[indexindex+1]+ A[indexindex-1])/4;
        }
}

__global__ void calculateKernelUpdate(float * A, float * B, ll n, ll iter) {

	ll i = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x * blockDim.x + threadIdx.x >= 4 * iter * (n-iter-1) ) return;

    float lf;
    float rr;
	float uu;
    float dd;

	i = revertMatrixTrans(n, iter, (blockIdx.x * blockDim.x + threadIdx.x));
        lf = zeroArea(i-1, n, iter) ? 0 : A[mediumTrans(n, iter, i - 1)];
        rr = zeroArea(i+1, n, iter) ? 0 : A[mediumTrans(n, iter, i + 1)];

        uu = zeroArea(i-(n + 1), n, iter) ? 0 : A[mediumTrans(n, iter, i - n - 1)];
        dd = zeroArea(i+(n + 1), n, iter) ? 0 : A[mediumTrans(n, iter, i + n + 1)];
        
	B[mediumTrans(n, iter, i)] = (lf + rr + uu + dd) / 4;
	
}
const int BS = 1024;

void calcDist(ll n, int iter, bool initialState) {

        ll i= (n+1)*(n+1)*sizeof(float);
        ll memsize = (n+1)*(n+1)*sizeof(float);

        float * H = (float*) calloc ((n+1)*(n+1), sizeof(float));

        for (i = 0; i < n+1; i++) {
                H[i] = (i >= 10 && i <= 30) ? 150 : 80;
                H[i*(n+1)] = 80;
                H[n*(n+1)+i] = 80;
                H[i*(n+1) + n] = 80;
        }

        float *Pd, *Qd;


        cudaMalloc((void**) &Qd, memsize);
        cudaMalloc((void**) &Pd, memsize);
        cudaMemcpy(Pd, H, memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(Qd, H, memsize, cudaMemcpyHostToDevice);


        for (i = 0; i < iter; i++) {
                if (i%2 == 0)
                        calculateKernel<<<ceil((double) (n-1)*(n-1)/BS), BS>>>(Pd, Qd, n+1);
                else
                        calculateKernel<<<ceil((double) (n-1)*(n-1)/BS), BS>>>(Qd, Pd, n+1);
        }


        cudaMemcpy(H, (i%2==1)?Qd:Pd , memsize, cudaMemcpyDeviceToHost);

        if(initialState) printSum(H,(n+1)*(n+1));
        //CLEAN UP
        cudaFree(Pd);
        cudaFree(Qd);

        free(H);
}

void calcDistOptimized(ll n, ll iter, bool initialState) {

	ll i = (4*n+4*iter*(n-iter-1))*sizeof(float);
    ll memsize = (4*n+4*iter*(n-iter-1))*sizeof(float);
	float * H = (float*) calloc ((4*n+4*iter*(n-iter-1)), sizeof(float));

	for (i = 0; i < n+1; i++) {
		H[transHost(n, iter, i)] = (i >= 10 && i <= 30) ? 150 : 80;
		H[transHost(n, iter, i * (n + 1))] = 80;
		H[transHost(n, iter, (n + 1)*n + i)] = 80;
		H[transHost(n, iter, i * (n + 1) + n)] = 80;
	}

	float *Pd, *Qd;
	
	cudaMalloc((void**) &Qd, memsize);
	cudaMalloc((void**) &Pd, memsize);
	cudaMemcpy(Pd, H, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(Qd, H, memsize, cudaMemcpyHostToDevice);

	for (i = 0; i < iter; i++) {
		if (i%2 == 0)
			calculateKernelUpdate<<<ceil((double) (4*iter*(n-iter-1))/BS), BS>>>(Pd, Qd, n, iter); 
		else
			calculateKernelUpdate<<<ceil((double) (4*iter*(n-iter-1))/BS), BS>>>(Qd, Pd, n, iter);
	}

	cudaMemcpy(H, (i%2==1)?Qd:Pd , memsize, cudaMemcpyDeviceToHost);

	if(initialState) printSum(H, n);
		
	cudaFree(Pd);
	cudaFree(Qd);

	free(H);
	
}
int main(int argc, char * argv[]) {
  
   //clock_t start = clock();
	ll n = 10000;
	ll iter = 500;

	bool initialState = false;
	if (argc > 1) n = atoi(argv[1]);
	if (argc > 2) iter = atoi(argv[2]);
	if (argc > 3) initialState = true;

	//dim3 dimGrid(
	int count, i;
	if (cudaGetDeviceCount(&count) != cudaSuccess) {
		printf("Get device count error!");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < count; i++) {
	}
	if ( n+1 > 2*(iter+1)) calcDistOptimized(n, iter, initialState);
	else calcDist(n, iter, initialState); 
	//clock_t end = (clock() - start)/1000;
  //printf("time: %ldms\n", end);
		
}

