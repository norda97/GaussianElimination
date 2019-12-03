#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_SIZE 4096
#define DEBUG false

typedef double matrix[MAX_SIZE][MAX_SIZE];

int			g_N;					/* Matrix size */
int			g_maxnum;				/* Max number of element */
const char* g_init;					/* Matrix init type	*/
bool		g_print;				/* Print switch	*/
bool		g_compare;				/* Compare switch*/
matrix		g_A;					/* Matrix A					- Sequential implementation	*/
double		g_b[MAX_SIZE];			/* Vector b					- Sequential implementation */
double		g_y[MAX_SIZE];			/* Vector y					- Sequential implementation */
int			g_threads;				/* Threads Requested		- Parallel implementation */
matrix		g_Matrix;				/* Matrix					- Parallel implementation */
double		g_result[MAX_SIZE];		/* Result Vector			- Parallel implementation */
bool		g_parallel;				/* Determens Implementation	- Parallel implementation */				
/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(matrix Matrix, double vecB[], double vecY[]);
void Init_Default(void);
void Read_Options(int, char**);
void compareImplementations();

cudaError_t gaussParallel();

/*
	Read only kernel memory
*/
__constant__ int d_size;
__constant__ int d_UtilThreads;

/*
	Debugging functions
*/
inline void handleError(const cudaError_t& status, int lineNumb) 
{
	if (status != cudaSuccess) {
		printf("[%d]Device failed due to %s: %s\n", lineNumb, cudaGetErrorName(status), cudaGetErrorString(status));
	}
}

__global__ void divStep(double* M, unsigned pivotIndex)
{
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	const int size = d_size;
	const int numThreads = d_UtilThreads;
	
	// Rows left until matrix is upper triangular
	const int newSize = size - pivotIndex;
	
	int workPerThread = ceilf((newSize+1) / (float)numThreads);
	
	double pivot = M[pivotIndex * (size + 1) + pivotIndex];
	int col = (pivotIndex+1) + tx;
	for (int i = 0; i < workPerThread; i++) {
		if (col < size+1)
			M[pivotIndex * (size+1) + col] /= pivot;
		col += numThreads;
	}
}

__global__ void elimStep(double* M, unsigned pivotIndex)
{
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	const int size = d_size;
	const int numThreads = d_UtilThreads;

	// Forward Elimantion
	int newSizeX = size - pivotIndex;
	int newSizeY = newSizeX - 1;

	int workPerThread = ceilf((newSizeX*newSizeY) / (float)numThreads);
	int index = tx;
	int col; int row; double factor;
	for (int i = 0; i < workPerThread; i++) {
		col = index % newSizeX;
		row = index / newSizeX;
		if (row >= newSizeY)
			break;

		// Get Factor which will be used to elimnate column of row
		row = (row + pivotIndex + 1) * (size + 1);
		factor = M[row + pivotIndex];

		// Eliminate
		col = col + (pivotIndex + 1);
		M[row + col] -= factor * M[pivotIndex * (size + 1) + col];

		index += numThreads;
	}

	// Backward Elimantion
	newSizeX = size - pivotIndex;
	newSizeY = pivotIndex;
	workPerThread = ceilf((newSizeX * newSizeY) / (float)numThreads);
	index = tx;
	for (int i = 0; i < workPerThread; i++) {
		col = index % newSizeX;
		row = index / newSizeX;
		if (row >= newSizeY)
			break;

		factor = M[row * (size + 1) + pivotIndex];
		M[row * (size + 1) + pivotIndex + 1 + col] -= factor * M[pivotIndex * (size + 1) + pivotIndex + 1 + col];
		index += numThreads;
	}
}

int main(int argc, char* argv[])
{

    // Create Cuda status
	cudaError_t cudaStatus;

	Init_Default();		/* Init default values	*/
	Read_Options(argc, argv);	/* Read arguments	*/
	Init_Matrix();		/* Init the matrix	*/

	if (g_compare) {
		work();
		gaussParallel();
	}
	else if (g_parallel) {
		gaussParallel();
		if (g_print == 1)
			Print_Matrix(g_Matrix, g_result, nullptr);
	}
	else {
		work();
		if (g_print == 1)
			Print_Matrix(g_A, g_b, g_y);
	}

	// Compare sequential and parallel result
	if (g_compare) {
		compareImplementations();
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void compareImplementations() {
	bool different = false;
	const double epsilon = 0.00001;
	for (int i = 0; i < g_N; i++) {
		if (fabsf(g_result[i] - g_y[i]) > epsilon) {
			different = true;
		}
	}
	if (different)
		printf("Compare: Parallel implementation differs!\n\n");
	else
		printf("Compare: Results are identical!\n\n");

	if (g_print == 1) {
		printf("SERIAL OUTPUT:\n");
		Print_Matrix(g_A, g_b, g_y);
		printf("PARALLEL OUTPUT:\n");
		Print_Matrix(g_Matrix, g_result, nullptr);
	}
}

void Init_Default() {
	g_N = 3;
	g_threads = 1;
	g_init = "rand";
	g_maxnum = 18.0;
	g_print = true;
	g_compare = true;
	g_parallel = true;
}

cudaError_t gaussParallel()
{
	double* d_Matrix = 0;
	cudaError_t cudaStatus;

	// Get System information
	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceProperties(&prop, 0);
	handleError(cudaStatus, __LINE__);

	// Allocate GPU buffer for the Matrix and the Result Vector
	int matrixAndResultSize = (g_N + 1);
	cudaStatus = cudaMalloc((void**)&d_Matrix, g_N * matrixAndResultSize * sizeof(double));
	handleError(cudaStatus, __LINE__);
	/*
	cudaStatus = cudaMalloc((void**)&d_VecResult, g_N * sizeof(double));
	handleError(cudaStatus, __LINE__);*/

	// Write length of Matrix and the maximum number of threads that will be utilized into constant memory
	cudaStatus = cudaMemcpyToSymbol(d_size, &g_N, sizeof(int), 0, cudaMemcpyHostToDevice);
	handleError(cudaStatus, __LINE__);
	cudaStatus = cudaMemcpyToSymbol(d_UtilThreads, &g_threads, sizeof(int), 0, cudaMemcpyHostToDevice);
	handleError(cudaStatus, __LINE__);

	// Copy Matrix and Vector from host memory to the same GPU buffer.
	for (int i = 0; i < g_N; i++) {
		int stride = i * (g_N + 1);
		cudaStatus = cudaMemcpy((d_Matrix + stride), g_Matrix[i], g_N * sizeof(double), cudaMemcpyHostToDevice);
		handleError(cudaStatus, __LINE__);

		// Copy Vector B to GPU
		cudaStatus = cudaMemcpy(d_Matrix + g_N + stride, (void*)&g_result[i], sizeof(double), cudaMemcpyHostToDevice);
		handleError(cudaStatus, __LINE__);
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadPerBlock(1, 1, 1);
	dim3 blocksGrid(1, 1, 1);
	const int maxThreadDim = prop.maxThreadsDim[0];
	if (g_threads > maxThreadDim) {
		threadPerBlock.x = maxThreadDim;
		const int numBlocks = ceilf(g_threads / (float)maxThreadDim);
		blocksGrid.x = numBlocks;
	}
	else {
		threadPerBlock.x = g_threads;
	}

	for (int i = 0; i < g_N; i++) {
		divStep << <blocksGrid, threadPerBlock >> > (d_Matrix, i);
		elimStep << <blocksGrid, threadPerBlock >> > (d_Matrix, i);
	}

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
	handleError(cudaStatus, __LINE__);

    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching gaussElim!\n", cudaStatus);
        goto Error;
    }

	// Copy Matrix and Vector from device buffer to Host.
	for (int i = 0; i < g_N; i++) {
		int stride = i * (g_N + 1);
		cudaStatus = cudaMemcpy(g_Matrix[i], (d_Matrix + stride), g_N * sizeof(double), cudaMemcpyDeviceToHost);
		handleError(cudaStatus, __LINE__);

		// Copy Result Vector to Host
		cudaStatus = cudaMemcpy(&g_result[i], (d_Matrix + g_N + stride), sizeof(double), cudaMemcpyDeviceToHost);
		handleError(cudaStatus, __LINE__);
	}

Error:
    cudaFree(d_Matrix);
    
    return cudaStatus;
}


void Read_Options(int argc, char** argv) {
	char* prog;

	prog = *argv;
	while (++argv, --argc > 0)
		if (**argv == '-')
			switch (*++ * argv) {
			case 'n':
				--argc;
				g_N = atoi(*++argv);
				break;
			case 'h':
				printf("\nUsage: sor [-n problemsize]\n");
				printf("           [-d] show default values \n");
				printf("           [-h] help \n");
				printf("           [-i init_type] fast/rand \n");
				printf("           [-m maxnum] max random no \n");
				printf("           [-p g_print] 0/1 \n");
				printf("           [-v g_parallel] 0/1 \n");
				printf("           [-c g_compare] 0/1 \n");
				printf("           [-t g_threads] Number of threads wanted \n");
				exit(0);
				break;
			case 'd':
				printf("\nDefault:  g_N(Matrix Size)			= %d ", g_N);
				printf("\n          g_init(Initialize type)		= %s", g_init);
				printf("\n          maxnum(Maximun Number)		= %d", g_maxnum);
				printf("\n          g_threads					= %d ",(int)g_threads);
				printf("\n          g_print(Print result)		= %d", (int)g_print);
				printf("\n          g_compare(Print Compare)	= %d \n\n", (int)g_compare);
				exit(0);
				break;
			case 'i':
				--argc;
				g_init = *++argv;
				break;
			case 'm':
				--argc;
				g_maxnum = atoi(*++argv);
				break;
			case 'p':
				--argc;
				g_print = atoi(*++argv);
				break;
			case 'c':
				--argc;
				g_compare = (bool)atoi(*++argv);
				break;
			case 'v':
				--argc;
				g_parallel = (bool)atoi(*++argv);
				break;
			case 't':
				--argc;
				g_threads = atoi(*++argv);
				break;
			default:
				printf("%s: ignored option: -%s\n", prog, *argv);
				printf("HELP: try %s -h \n\n", prog);
				break;
			}
}

void work(void)
{
	int i, j, k;

	/* Gaussian elimination algorithm, Algo 8.4 from Grama */
	for (k = 0; k < g_N; k++) { /* Outer loop */
		for (j = k + 1; j < g_N; j++) {
			g_A[k][j] = g_A[k][j] / g_A[k][k]; /* Division step */
		}
		g_y[k] = g_b[k] / g_A[k][k];
		g_A[k][k] = 1.0;

		for (i = k + 1; i < g_N; i++) {
			for (j = k + 1; j < g_N; j++) {
				g_A[i][j] = g_A[i][j] - g_A[i][k] * g_A[k][j]; /* Elimination step */

			}
			g_b[i] = g_b[i] - g_A[i][k] * g_y[k];
			g_A[i][k] = 0.0;
		}
	}	
}

void Init_Matrix() {
	int i, j;
	if (g_print) {
		printf("\nMatrix Size      = %dx%d ", g_N, g_N);
		printf("\nMaximum Number    = %d \n", g_maxnum);
		printf("Initialize type		= %s \n", g_init);
		printf("Threads				= %d \n", g_threads);
		printf("Initializing matrix...\n\n");
		if(g_parallel)
			printf("Parallel Implementation:\n");
		else
			printf("Serial Implementation:\n");
	}
	if (strcmp(g_init, "rand") == 0) {
		for (i = 0; i < g_N; i++) {
			for (j = 0; j < g_N; j++) {
				if (i == j) /* diagonal dominance */
					g_A[i][j] = (double)(rand() % g_maxnum) + 5.0;
				else
					g_A[i][j] = (double)(rand() % g_maxnum) + 1.0;

				g_Matrix[i][j] = g_A[i][j];
			}
		}
	}
	if (strcmp(g_init, "fast") == 0) {
		for (i = 0; i < g_N; i++) {
			for (j = 0; j < g_N; j++) {
				if (i == j) /* diagonal dominance */
					g_A[i][j] = 5.0;
				else
					g_A[i][j] = 2.0;

				g_Matrix[i][j] = g_A[i][j];
			}
		}
	}

	/* Initialize vectors b and y */
	for (i = 0; i < g_N; i++) {
		g_b[i] = 2.0;
		g_result[i] = g_b[i];
		g_y[i] = 1.0;
	}

	if (g_print == 1)
		Print_Matrix(g_A, g_b, g_y);
}

void Print_Matrix(matrix Matrix, double vecB[], double vecY[]) {
	int i, j;

	printf("Matrix A:\n");
	for (i = 0; i < g_N; i++) {
		printf("[");
		for (j = 0; j < g_N; j++)
			printf(" %5.2f,", Matrix[i][j]);
		printf("]\n");
	}
	if (vecY != nullptr) {
		printf("Vector b:\n[");
		for (j = 0; j < g_N; j++)
			printf(" %5.2f,", vecB[j]);
		printf("]\n");

		printf("Vector y:\n[");
		for (j = 0; j < g_N; j++)
			printf(" %5.2f,", vecY[j]);
		printf("]\n");
	}
	else {
		printf("Vector result:\n[");
		for (j = 0; j < g_N; j++)
			printf(" %5.2f,", vecB[j]);
		printf("]\n");
	}
	
	printf("\n\n");
}
