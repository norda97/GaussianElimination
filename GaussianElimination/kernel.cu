/*****************************************************
*
* Gaussian elimination
*
* Sequential version
*
*****************************************************/
// Compile and then...
// Example run 1:   gauseq.exe -P 1 -I fast -n 16
// Example run 2:   gauseq.exe -P 0 -I rand -n 2048
// 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 2048

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* matrix size		*/
int	maxnum;		/* max number of element*/
const char* Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */

						/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char**);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char* argv[])
{

    // Create Cuda status
	cudaError_t cudaStatus;

	int i, timestart, timeend, iter;

	Init_Default();		/* Init default values	*/
	Read_Options(argc, argv);	/* Read arguments	*/
	Init_Matrix();		/* Init the matrix	*/
	work();
	if (PRINT == 1)
		Print_Matrix();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void Init_Default() {
	N = 32;
	Init = "rand";
	maxnum = 15.0;
	PRINT = 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


void work(void)
{
	int i, j, k;

	/* Gaussian elimination algorithm, Algo 8.4 from Grama */
	for (k = 0; k < N; k++) { /* Outer loop */
		for (j = k + 1; j < N; j++)
			A[k][j] = A[k][j] / A[k][k]; /* Division step */
		y[k] = b[k] / A[k][k];
		A[k][k] = 1.0;
		for (i = k + 1; i < N; i++) {
			for (j = k + 1; j < N; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j]; /* Elimination step */
			b[i] = b[i] - A[i][k] * y[k];
			A[i][k] = 0.0;
		}
	}
}

void Init_Matrix() {
	int i, j;

	printf("\nsize      = %dx%d ", N, N);
	printf("\nmaxnum    = %d \n", maxnum);
	printf("Init	  = %s \n", Init);
	printf("Initializing matrix...");

	if (strcmp(Init, "rand") == 0) {
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				if (i == j) /* diagonal dominance */
					A[i][j] = (double)(rand() % maxnum) + 5.0;
				else
					A[i][j] = (double)(rand() % maxnum) + 1.0;
			}
		}
	}
	if (strcmp(Init, "fast") == 0) {
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				if (i == j) /* diagonal dominance */
					A[i][j] = 5.0;
				else
					A[i][j] = 2.0;
			}
		}
	}

	/* Initialize vectors b and y */
	for (i = 0; i < N; i++) {
		b[i] = 2.0;
		y[i] = 1.0;
	}

	printf("done \n\n");
	if (PRINT == 1)
		Print_Matrix();
}

void Print_Matrix() {
	int i, j;

	printf("Matrix A:\n");
	for (i = 0; i < N; i++) {
		printf("[");
		for (j = 0; j < N; j++)
			printf(" %5.2f,", A[i][j]);
		printf("]\n");
	}
	printf("Vector b:\n[");
	for (j = 0; j < N; j++)
		printf(" %5.2f,", b[j]);
	printf("]\n");
	printf("Vector y:\n[");
	for (j = 0; j < N; j++)
		printf(" %5.2f,", y[j]);
	printf("]\n");
	printf("\n\n");
}

int Read_Options(int argc, char** argv) {
	char* prog;

	prog = *argv;
	while (++argv, --argc > 0)
		if (**argv == '-')
			switch (*++ * argv) {
			case 'n':
				--argc;
				N = atoi(*++argv);
				break;
			case 'h':
				printf("\nHELP: try sor -u \n\n");
				exit(0);
				break;
			case 'u':
				printf("\nUsage: sor [-n problemsize]\n");
				printf("           [-D] show default values \n");
				printf("           [-h] help \n");
				printf("           [-I init_type] fast/rand \n");
				printf("           [-m maxnum] max random no \n");
				printf("           [-P print_switch] 0/1 \n");
				exit(0);
				break;
			case 'D':
				printf("\nDefault:  n         = %d ", N);
				printf("\n          Init      = rand");
				printf("\n          maxnum    = 5 ");
				printf("\n          P         = 0 \n\n");
				exit(0);
				break;
			case 'I':
				--argc;
				Init = *++argv;
				break;
			case 'm':
				--argc;
				maxnum = atoi(*++argv);
				break;
			case 'P':
				--argc;
				PRINT = atoi(*++argv);
				break;
			default:
				printf("%s: ignored option: -%s\n", prog, *argv);
				printf("HELP: try %s -u \n\n", prog);
				break;
			}
}
