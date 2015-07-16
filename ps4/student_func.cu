//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <cstdio>
#include <algorithm>
#define makePow2(v) \
	v--;			\
	v |= v >> 1;	\
	v |= v >> 2;	\
	v |= v >> 4;	\
	v |= v >> 8;	\
	v |= v >> 16;	\
	v++;			\


/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
 
#define lognbins 2
#define numBins (1<<lognbins)

__global__
void histogram(const unsigned int* const d_in, unsigned int* const d_out, 
				const size_t v, int pad, int size)
{
	unsigned int l_bins[numBins];
	memset(l_bins, 0, sizeof(int)*numBins);
	int bid=blockIdx.x+gridDim.x*blockIdx.y+gridDim.x*gridDim.y*blockIdx.z;
	int id=bid*blockDim.x+threadIdx.x;
	if(id>=size)
		return;
	
	for(int i=0;i<v;i++)
	{
		if(id*v+i>=size)
			break;
		l_bins[(d_in[id*v+i]>>pad)&(numBins-1)]++;
	}
		
	for(int i=0;i<numBins;i++)
		atomicAdd(d_out+i, l_bins[i]);
}

/*Hillis and Steele Exclusive Scan*/
__global__
void scan(const unsigned int* const d_in, unsigned int* const d_out,
			unsigned int* const d_blockout, int n)
{
	extern __shared__ unsigned int l_mod[];
	
	int bid=blockIdx.x+gridDim.x*blockIdx.y+gridDim.x*gridDim.y*blockIdx.z;
	int id=bid*blockDim.x+threadIdx.x;
	int tid=threadIdx.x;
	unsigned int tmp;
	
	if(id>=n || bid>n/blockDim.x)
		return;
	l_mod[tid] = d_in[id];
	__syncthreads();
	
	for(unsigned int s=1;s<blockDim.x;s<<=1)
	{
		tmp=l_mod[tid];
		__syncthreads();
		
		if(tid+s<blockDim.x)
			l_mod[tid+s]+=tmp;
		__syncthreads();
	}
	d_out[id] = tid>0?l_mod[tid-1]:0;
	if(tid==1023)
		d_blockout[bid]=l_mod[tid];
}

__global__
void add(const unsigned int* const d_in, unsigned int* const d_out, 
			const unsigned int* d_incr, int n)
{
	int bid=blockIdx.x+gridDim.x*blockIdx.y+gridDim.x*gridDim.y*blockIdx.z;
	int id=bid*blockDim.x+threadIdx.x;
	if(id>=n)
		return;
	d_out[id]=d_in[id]+d_incr[bid];
}

/*do scan for HUUUUGE ARRAYS*/
void prefixSum(unsigned int* d_arr, unsigned int* d_out, int size)
{
	unsigned int* d_blockout;
	int n=size;
	makePow2(n);
	int threads=1024;
	dim3 blocks(n/threads,n/threads/65535+1,n/threads/65535/65535+1);
	blocks.x=min(blocks.x, 65535);
	blocks.y=min(blocks.y, 65535);
	blocks.z=min(blocks.z, 65535);
	blocks.x=max(1, blocks.x);
	checkCudaErrors(cudaMalloc(&d_blockout, sizeof(unsigned int)*blocks.x*blocks.y*blocks.z));
	scan<<<blocks, threads, threads*sizeof(unsigned int)>>>(d_arr, d_out, d_blockout, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	if(blocks.x>1)
	{
		prefixSum(d_blockout, d_blockout, blocks.x*blocks.y*blocks.z);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		add<<<blocks, threads>>>(d_out, d_out, d_blockout, size);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	cudaFree(d_blockout);
}

__global__
void map(unsigned int* d_in, unsigned int* d_out, const size_t size, int mask, int pad)
{
	int bid=blockIdx.x+gridDim.x*blockIdx.y+gridDim.x*gridDim.y*blockIdx.z;
	int id=bid*blockDim.x+threadIdx.x;
	
	if(id>=size)
		return;
	d_out[id] = ((d_in[id]>>pad)&(numBins-1))==mask;
}

__global__
void move(unsigned int* const d_inputVals, unsigned int* const d_inputPos,
			unsigned int* const d_outputVals, unsigned int* const d_outputPos,
			unsigned int* const d_pred, unsigned int* d_scan, 
			unsigned int* offset, const size_t size, int mask)
{
	int bid=blockIdx.x+gridDim.x*blockIdx.y+gridDim.x*gridDim.y*blockIdx.z;
	int id=bid*blockDim.x+threadIdx.x;
	if(id>=size || !d_pred[id])
		return;
	int outp = offset[mask]+d_scan[id];
	d_outputVals[outp] = d_inputVals[id];
	d_outputPos[outp] = d_inputPos[id];
}

void your_sort(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               const size_t numElems)
{
	unsigned int n=numElems;
	makePow2(n);
	const int histSerialization=64;
	int threads=1024;
	unsigned int* d_hist, *d_pred, *d_scan;
	dim3 blocksHist(n/threads/histSerialization, n/threads/histSerialization/65535+1,
					n/threads/histSerialization/65535/65535+1);
	blocksHist.x=min(blocksHist.x, 65535);
	blocksHist.y=min(blocksHist.y, 65535);
	blocksHist.z=min(blocksHist.z, 65535);
	blocksHist.x=max(1, blocksHist.x);
	dim3 blocksMap(n/threads, n/threads/65535+1, n/threads/65535/65535+1);
	blocksMap.x=min(blocksMap.x, 65535);
	blocksMap.y=min(blocksMap.y, 65535);
	blocksMap.z=min(blocksMap.z, 65535);
	blocksMap.x=max(1, blocksMap.x);
	
	checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int)*numBins));
	checkCudaErrors(cudaMalloc(&d_pred, sizeof(unsigned int)*numElems));
	checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int)*numElems));
	for(int i=0;i<8*(int)sizeof(unsigned int);i+=lognbins)
	{
		checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int)*numBins));
		histogram<<<blocksHist, threads>>>(d_inputVals, d_hist, histSerialization, i, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		prefixSum(d_hist, d_hist, numBins);
		for(int j=0;j<numBins;j++)
		{
			map<<<blocksMap, threads>>>(d_inputVals, d_pred, numElems, j, i);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			prefixSum(d_pred, d_scan, numElems);
			move<<<blocksMap, threads>>>(d_inputVals, d_inputPos, d_outputVals, 
											d_outputPos, d_pred, d_scan, d_hist, 
											numElems, j);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}
		std::swap(d_inputPos, d_outputPos);
		std::swap(d_inputVals, d_outputVals);
	}
	cudaMemcpy(d_outputVals, d_inputVals, numElems*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos, d_inputPos, numElems*sizeof(int), cudaMemcpyDeviceToDevice);
}

