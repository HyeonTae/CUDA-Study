#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define N 100
__global__ void mul(int a[][N], int b[][N], int c[][N]){
	int row = blockIdx.x;
	int col = blockIdx.y;

	if(row < N && col < N) {
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				c[i][j] = 0;
				for(int k = 0; k < N; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
}

int main(){
	int (*pa)[N], (*pb)[N], (*pc)[N];
	int a[N][N], b[N][N], c[N][N];

	srand((unsigned)time(NULL));	

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
	
	cudaMalloc((void**)&pa, (N*N) * sizeof(int));
	cudaMalloc((void**)&pb, (N*N) * sizeof(int));
	cudaMalloc((void**)&pc, (N*N) * sizeof(int));

	for(int i = 0 ; i<N ; i++){
		for(int j = 0 ; j<N ; j++) {
			a[i][j] = rand()%10 + 1;
			b[i][j] = rand()%10 + 1;
		}
	}
	
	cudaMemcpy(pa, a, (N*N) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pb, b, (N*N) * sizeof(int), cudaMemcpyHostToDevice);	
	
	dim3 blocksPerBlock(N,N);
	mul<<<blocksPerBlock,1>>>(pa, pb, pc);

	cudaMemcpy(c, pc, (N*N) * sizeof(int), cudaMemcpyDeviceToHost);

	printf("matrix multiplication per block\n");

/*	for(int i = 0 ; i<N ; i++){
		for(int j = 0 ; j<N ; j++) {
			printf("%d ",c[i][j]);
		}
		printf("\n");
	}
*/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Time to generate : %3.1f ms\n", elapsedTime);

	cudaFree(pa);
	cudaFree(pb);
	cudaFree(pc);

	return 0;

}
