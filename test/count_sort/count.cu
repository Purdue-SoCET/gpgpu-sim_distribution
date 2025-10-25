#include "count.h"

int main(int argc, const char **argv) {

	if (argc != 2) {
		printf("Please provide an array size, size = 2^k. Provide k as an argument to the program.\n");
		exit(EXIT_FAILURE);
	}

	int n = pow(2, atoi(argv[1])),
		A[n], B[MAX_VALUE], C[n],
		*dA, *dB;

	// malloc device memory
	cudaMalloc((void**)&dA, sizeof(int) * n);
	cudaMalloc((void**)&dB, sizeof(int) * BLOCKS * MAX_VALUE);

	// initialize srand for randomizing values based on time.
	time_t time;
	srand((time)&time);

	// dump random values in array A.
	for (int i = 0; i < n; ++i)
		A[i] = rand() % MAX_VALUE;

	// copy the content of array A to dA on the device.
	cudaMemcpy(dA, A, sizeof(int) * n, cudaMemcpyHostToDevice);

	// call the count kernal.
	count<<<BLOCKS, THREADS>>>(dA, dB, n);

	// merge count ararys using the merge kernal.
	merge<<<BLOCKS, THREADS>>>(dB);

	// copy result of the summation of counted arrays to host.
	cudaMemcpy(B, dB, sizeof(int) * MAX_VALUE, cudaMemcpyDeviceToHost);

	// construct the sorted array.
	for (int i = 0, j = 0; i < MAX_VALUE; ++i)
		for (int k = 0; k < B[i]; ++k, ++j)
			C[j] = i;

	// print unsorted and sorted arrays.
	print("Original", A, n);
	print("Sorted", C, n);

	// free device memory.
	cudaFree(dA);
	cudaFree(dB);

	return EXIT_SUCCESS;

}

__global__ void count(int *A, int *B, int n) {

	int b_id 		= blockIdx.x,
			b_num 	= gridDim.x,
			b_size,
			b_offset,
			t_id 	= threadIdx.x,
			t_num 	= blockDim.x,
			t_size,
			t_offset,
			offset;

	// initialize a shared memory array to store the count for each block.
	__shared__ int count[MAX_VALUE];

	// set intial values to zeros. Each thread sets its own share to zero.
	t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
	offset = t_id * t_size;
	for (int i = offset; i < offset + t_size && i < MAX_VALUE; ++i)
		count[i] = 0;

	// wait until all threads have completed the initialization process.
	__syncthreads();

	// accumulate the counts of each value. Each thread counts a certain portain
	// of the unsorted array.
	b_size = (b_num > n ? 1 : n / b_num);
	b_offset = b_id * b_size;

	t_size = (t_num > b_size ? 1 : b_size / t_num);

	offset = b_offset + t_id * t_size;
	for (int i = offset; i < offset + t_size && i < b_offset + b_size && i < n; ++i)
		atomicAdd(&count[A[i]], 1);

	// wait until all threads have completed the couting phase.
	__syncthreads();

	// copy the block count into global memory. Each thread copies its portioin to 
	// the global memory.
	t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
	t_offset = t_id * t_size;
	offset = b_id * MAX_VALUE + t_offset;

	if (offset + t_size <= (b_id + 1) * MAX_VALUE)
		memcpy(&B[offset], &count[t_offset], sizeof(int) * t_size);

}

__global__ void merge(int *B) {

	int b_id	= blockIdx.x,
		b_num	= gridDim.x,
		b_size,
		b_offset,
		t_id	= threadIdx.x,
		t_num	= blockDim.x,
		t_size,
		offset;

	// loop through and merge until all arrays are merged.
	for (int i = b_num, j = 2; i != 1; i /= 2, j *= 2) {

		// each block will operate on b_size values which equal, the number of 
		// count arrays * size of count arrays / number of blocks / 2. The final 2
		// represents the merge process.
		b_size = i * MAX_VALUE / b_num / 2;
		b_offset = (b_id / j) * (j * MAX_VALUE) + b_size * (b_id % j);

		t_size = (t_num > b_size ? 1 : b_size / t_num);

		// calculate the offset that each thread will start at and sum counts.
		offset = b_offset + t_id * t_size;
		for (int k = offset, l = offset + (MAX_VALUE * (j / 2)); 
			k < offset + t_size && k < b_offset + b_size; ++k, ++l)
			B[k] += B[l];

		// wait untill all arrays are merged for every step.
		__syncthreads();

	}

}

// print results.
void print(const char *name, int *array, int size) {

	printf("%s = [%d", name, array[0]);
	for (int i = 1; i < size; ++i) printf(", %d", array[i]);
	printf("]\n");

}