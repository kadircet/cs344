/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

typedef float(*reduce_func)(const float, const float);

__device__
float getMax(const float a, const float b)
{
	return a>b?a:b;
}

__device__
float getMin(const float a, const float b)
{
	return a>b?b:a;
}

__device__ reduce_func d_getMax = getMax;
__device__ reduce_func d_getMin = getMin;

__global__
void reduce(const float* const d_in, float* const d_out, size_t v, 
				reduce_func op)
{
	extern __shared__ float sdata[];
	int id = blockDim.x*blockIdx.x+threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = d_in[id];
	__syncthreads();
	
	/*to make size a power of 2*/
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	/*v is the smallest power of 2 that is larger than the number of elements*/
	
	for(unsigned int s=v/2;s>0;s>>=1)
	{
		if(tid<s && tid+s<blockDim.x)
			sdata[tid] = op(sdata[tid], sdata[tid+s]);
		__syncthreads();
	}
	if(tid==0)
		d_out[blockIdx.x]=sdata[0];
}

__global__
void histogram(const float* const d_in, unsigned int* const d_out, const size_t numBins,
				const float minlogLum, const float lumRange, const size_t v)
{
	unsigned int *l_bins = new unsigned int[numBins];
	memset(l_bins, 0, sizeof(int)*numBins);
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	
	for(int i=0;i<v;i++)
	{
		int bin = (d_in[id*v+i]-minlogLum)/lumRange*numBins;
		if(bin>=numBins)
			bin=numBins-1;
		l_bins[bin]++;
	}
	for(unsigned int i=0;i<numBins;i++)
		atomicAdd(d_out+i, l_bins[i]);
}

/*Hillis and Steele Scan*/
__global__
void scan(const unsigned int* const d_in, unsigned int* const d_out, const size_t numBins)
{
	extern __shared__ unsigned int l_mod[];
	
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int tid=threadIdx.x;
	
	l_mod[tid] = d_in[id];
	__syncthreads();
	
	for(unsigned int s=1;s<numBins;s<<=1)
	{
		if(tid+s<blockDim.x)
			l_mod[tid+s]+=l_mod[tid];
		__syncthreads();
	}
	d_out[id] = tid>0?l_mod[tid-1]:0;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	int threads	= 1024,
		blocks	= numRows*numCols/1024;
	float *d_inter, *d_out, lumRange;
	unsigned int *d_hist;
	reduce_func h_getMax, h_getMin;
	
	checkCudaErrors(cudaMemcpyFromSymbol(&h_getMax, d_getMax, sizeof(reduce_func)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_getMin, d_getMin, sizeof(reduce_func)));
	checkCudaErrors(cudaMalloc(&d_inter, sizeof(float) * blocks));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(float) * 1));
	checkCudaErrors(cudaMalloc(&d_hist, sizeof(int) * numBins));
	checkCudaErrors(cudaMemset(d_hist, 0, sizeof(int)*numBins));
	checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(int)*numBins));
	
	reduce<<<blocks, threads, threads*sizeof(float)>>>(d_logLuminance, d_inter, threads, h_getMax);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	threads=blocks;
	blocks=1;
	reduce<<<blocks, threads, threads*sizeof(float)>>>(d_inter, d_out, threads, h_getMax);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	
	threads	= 1024;
	blocks	= numRows*numCols/1024;
	reduce<<<blocks, threads, threads*sizeof(float)>>>(d_logLuminance, d_inter, threads, h_getMin);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	threads=blocks;
	blocks=1;
	reduce<<<blocks, threads, threads*sizeof(float)>>>(d_inter, d_out, threads, h_getMin);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&min_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
		
	lumRange = max_logLum - min_logLum;
	
	#define v 256
	threads = 1024/v;
	blocks	= numRows*numCols/1024;
	histogram<<<blocks, threads>>>(d_logLuminance, d_hist, numBins, min_logLum, 
									lumRange, v);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	blocks=1;
	threads=numBins/blocks;
	scan<<<blocks, threads, threads*sizeof(unsigned int)>>>(d_hist, d_cdf, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	checkCudaErrors(cudaFree(d_hist));
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_inter));
}
