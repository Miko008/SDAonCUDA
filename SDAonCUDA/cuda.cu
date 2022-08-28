#include <cuda_runtime.h>
#include <cmath>

#include "device_launch_parameters.h"
#include "main.h"
#include "cuda.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace GPU
{
	__global__ void SDAKernel3D(uint8_t* in, uint8_t* out, uint32_t frames, uint32_t height, uint32_t width, float rsqr, uint16_t iradius, int threshold, uint64_t size)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
		uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x > width || y > height || z > frames)
			return;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (0 <= x + i && x + i < width && i * i + j * j + k * k <= rsqr)
								if (in[((z + k) * height + y + j) * width + x + i] >= in[(z * height + y) * width + x] + threshold)
									out[(z * height + y) * width + x]++;
	}
	__global__ void SDAKernel1D(uint8_t* in, uint8_t* out, uint32_t frames, uint32_t height, uint32_t width, float rsqr, uint16_t iradius, int threshold, uint64_t size)
	{
		//todo omit using division operations
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tempid >= size)
			return;
 		uint32_t x = (tempid) % width;
		tempid /= width;
		uint32_t y = tempid % height;
		tempid /= height;
		uint32_t z = tempid % frames;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (0 <= x + i && x + i < width && i * i + j * j + k * k <= rsqr)
								if (in[((z + k) * height + y + j) * width + x + i] >= in[(z * height + y) * width + x] + threshold)
									out[(z * height + y) * width + x]++;
	}

	//Todo fix arg templates - linker error
	//template<class InBitDepth, class OutBitDepth>
	void SDA(Image<uint8_t> &input, Image<uint8_t> &output, float radius, int threshold)
	{
		uint8_t* devInput,* devOutput;
		uint64_t size = input.GetSize();
		cudaMalloc((void**)&devInput,  size * sizeof(uint8_t));
		cudaMalloc((void**)&devOutput, size * sizeof(uint8_t));

		cudaMemcpy(devInput, input.GetDataPtr(), size * sizeof(uint8_t), cudaMemcpyHostToDevice);

		uint16_t iradius = std::ceil(radius);

		//dim3 numBlocks(64, 8, 8);
		//dim3 threadsPerBlock(8, 8, 8);
		//SDAKernel3D<<<numBlocks, threadsPerBlock>>>(devInput, devOutput, frames, height, width, radius, iradius, threshold, size);
		
		dim3 numBlocks(size / 1024 + 1, 1, 1);
		dim3 threadsPerBlock(1024, 1, 1);
		SDAKernel1D<<<numBlocks, threadsPerBlock>>>
			(devInput, devOutput, input.Frames(), input.Height(), input.Width(), radius * radius, iradius, threshold, size);

		cudaDeviceSynchronize();

		cudaMemcpy(output.GetDataPtr(), devOutput, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

		cudaFree(devInput);
		cudaFree(devOutput);
	}

	__device__ uint8_t CalculateDominanceOverMoreIntense(int intensity, uint8_t* histogram, uint16_t diffLen)
	{
		uint16_t result = 0;
		uint32_t end = intensity > diffLen ? diffLen : intensity;    //select end value not higher than diffLen

		for (uint32_t i = 0; i < end; i++)    //add numbers of pixels that are >= pixel + threshold
			result += histogram[i];

		return result;
	}


	__device__ uint8_t CalculateDominanceOverLessIntense(int intensity, uint8_t* histogram, uint16_t diffLen)
	{
		uint16_t result = 0;
		uint32_t start = intensity > 0 ? intensity : 0;

		for (uint32_t i = start; i < diffLen; i++)			//add numbers of pixels that are >= pixel + threshold
			result += histogram[i];

		return result;
	}


	__global__ void FHFirstHistogramKernel(uint8_t* in, uint8_t* histogram, uint16_t histogramWidth, uint32_t frames, uint32_t height, uint32_t width, float radius, uint16_t iradius, int threshold)
	{
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t x = tempid % width;

		if (tempid < iradius || tempid > width - iradius)		//skip margins
			return;

		float asqr = radius * radius;

		for (int16_t k = -iradius; k <= iradius; k++)
			for (int16_t j = -iradius; j <= iradius; j++)
				for (int16_t i = -iradius; i <= iradius; i++)
					if (i * i + j * j + k * k <= asqr)
						histogram[histogramWidth * x +                                  //number of histogram
						in[((iradius + k) * height + iradius + j) * width + x + i]]++;		//value of intensity to histogram 
	}

	__device__ uint32_t CalculateIndex(Coords &Diff, uint32_t z, uint32_t y, uint32_t x, uint32_t height, uint32_t width)
	{
		return ((z + Diff.z) * height + y + Diff.y) * width + x + Diff.x;
	}

	__global__ void FHKernel(uint8_t* in, uint8_t* out, uint8_t* histogram, uint8_t* histogramCopy, uint8_t* histogramCopy2, uint16_t histogramWidth, Coords* DiffRemZ, Coords* DiffAddZ, Coords* DiffRemY, Coords* DiffAddY, uint16_t diffLen, uint32_t frames, uint32_t height, uint32_t width, uint16_t iradius, int threshold, bool moreIntense)
	{
		//todo omit using division operations
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t x = tempid % width;

		if (x < iradius || tempid > width - iradius)		//skip margins
			return;

		auto CalculateDominance = moreIntense ? CalculateDominanceOverLessIntense :
			CalculateDominanceOverMoreIntense;
		
		for (uint32_t z = iradius; z < frames - iradius; z++)
		{
			if (z != iradius)
			{
				for (uint32_t i = 0; i < diffLen; i++)      // compute by removing and adding delta pixels to histogram
				{
					histogram[histogramWidth * x +
						in[CalculateIndex(DiffRemZ[i], z, iradius, x, height, width)]]--;
					histogram[histogramWidth * x +
						in[CalculateIndex(DiffAddZ[i], z, iradius, x, height, width)]]++;
				}
			}

			for (size_t i = 0; i < histogramWidth * width; i++)
				histogramCopy[i] = histogram[i];

			for (size_t i = 0; i < histogramWidth; i++)
				histogramCopy2[i] = histogram[i + histogramWidth * x];
			
			for (uint32_t y = iradius; y < height - iradius; y++)
			{
				if (y != iradius)
				{
					for (uint32_t i = 0; i < diffLen; i++)      // compute by removing and adding delta pixels to histogram
					{
						histogramCopy[histogramWidth * x +
							in[CalculateIndex(DiffRemY[i], z, y, x, height, width)]]--;
						histogramCopy[histogramWidth * x +
							in[CalculateIndex(DiffAddY[i], z, y, x, height, width)]]++;
						/*histogramCopy2[
							in[CalculateIndex(DiffRemY[i], z, y, x, height, width)]]--;
						histogramCopy2[
							in[CalculateIndex(DiffAddY[i], z, y, x, height, width)]]++;*/
					}
				}

				uint16_t result = 0;
				uint16_t intensity = in[(z * height + y) * width + x] + threshold;
				
				if (!moreIntense)
				{
					for (uint64_t i = histogramWidth * x;
						i < histogramWidth * x + intensity; i++)		//from [x][0] to [x][intensity-1]
						result += histogramCopy[i];
				}
				else
				{
					for (uint64_t i = histogramWidth * x + intensity;
						i < histogramWidth * (x + 1); i++)				//from [x][intensity] to [x][max]
						result += histogramCopy[i];
				}

				//if (!moreIntense)
				//{
				//	for (uint64_t i = 0;
				//		i < intensity; i++)
				//		result += histogramCopy2[i];
				//}
				//else
				//{
				//	for (uint64_t i = intensity;
				//		i < histogramWidth; i++)				//from [x][intensity] to [x][max]
				//		result += histogramCopy2[i];
				//}

				//uint16_t* result = new uint16_t;\
				CalculateDominance(in[(z * height + y) * width + x] + threshold, \
					histogram, \
					diffLen, \
					histogramWidth * x, \
					result);

				out[(z * height + y) * width + x] = result;

			}
		}
	}

	void FlyingHistogram(Image<uint8_t> &input, Image<uint8_t> &output, float radius, int threshold, bool moreIntense)
	{

		uint16_t iradius = std::ceil(radius);

		uint16_t DiffLen = 0, DiffLenZ = 0;
		Coords* DiffAddZ, * DiffRemZ, * DiffAddY, * DiffRemY;//, * DiffAddX, * DiffRemX;   //array of coords of delta pixels

		DiffLenZ =	SetUpRadiusDifference(radius, &DiffAddZ, &DiffRemZ, true, Direction::Z); //number of delta pixels
		DiffLen  =	SetUpRadiusDifference(radius, &DiffAddY, &DiffRemY, true, Direction::Y);
		//			SetUpRadiusDifference(radius, &DiffAddX, &DiffRemX, true, Direction::X);

		//to do anisotropic
		//float asqr = radius * radius;
		//float csqr = radiusZ * radiusZ;


		dim3 numBlocks(input.Width() / 1024 + 1, 1, 1);
		dim3 threadsPerBlock(1024, 1, 1);


		uint64_t size = input.GetSize();
		uint8_t* devInput, * devOutput;

		Coords* devDiffAddZ, * devDiffRemZ, * devDiffAddY, * devDiffRemY;
		gpuErrchk(cudaMalloc(&devDiffAddZ, DiffLenZ * sizeof(Coords)));
		gpuErrchk(cudaMalloc(&devDiffRemZ, DiffLenZ * sizeof(Coords)));
		gpuErrchk(cudaMalloc(&devDiffAddY, DiffLen  * sizeof(Coords)));
		gpuErrchk(cudaMalloc(&devDiffRemY, DiffLen  * sizeof(Coords)));
		gpuErrchk(cudaMemcpy(devDiffAddZ, DiffAddZ, DiffLenZ * sizeof(Coords), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(devDiffRemZ, DiffRemZ, DiffLenZ * sizeof(Coords), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(devDiffAddY, DiffAddY, DiffLen * sizeof(Coords), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(devDiffRemY, DiffRemY, DiffLen * sizeof(Coords), cudaMemcpyHostToDevice));

		uint32_t frames = input.Frames(),
				 height = input.Height(),
				 width  = input.Width();

		uint8_t* devHistogram;						//array of starter histograms - only for device memory, no need to copy to host
		uint8_t* devHistogramCopy;
		uint8_t* devHistogramCopy2;
		uint16_t histogramWidth = std::numeric_limits<uint8_t>::max() + 1;
		uint64_t histogramSize = histogramWidth * width * sizeof(uint8_t);

		cudaMalloc(&devHistogram, histogramSize);
		cudaMalloc(&devHistogramCopy, histogramSize);
		cudaMalloc(&devHistogramCopy2, histogramWidth * sizeof(uint8_t));
		cudaMalloc(&devInput,  size * sizeof(uint8_t));
		cudaMalloc(&devOutput, size * sizeof(uint8_t));

		gpuErrchk(cudaMemcpy(devInput, input.GetDataPtr(), size * sizeof(uint8_t), cudaMemcpyHostToDevice));
		
		FHFirstHistogramKernel<<<numBlocks, threadsPerBlock>>>
			(devInput, devHistogram, histogramWidth, frames, height, width, radius, iradius, threshold);

		cudaDeviceSynchronize();

		FHKernel<<<numBlocks, threadsPerBlock>>>
			(devInput, devOutput, devHistogram, devHistogramCopy, devHistogramCopy2, histogramWidth, devDiffRemZ, devDiffAddZ, devDiffRemY, devDiffAddY,
				DiffLen, frames, height, width, iradius, threshold, moreIntense);

		cudaDeviceSynchronize();

		//uint8_t* testOut = new uint8_t[size];
		//memset(testOut, 0, size);
		//gpuErrchk(cudaMemcpy(testOut, devOutput, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(output.GetDataPtr(), devOutput, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

		//memcpy(output.GetDataPtr(), testOut, size * sizeof(uint8_t));

		cudaFree(devDiffAddZ);
		cudaFree(devDiffRemZ);
		cudaFree(devDiffAddY);
		cudaFree(devDiffRemY);

		cudaFree(devHistogramCopy2);
		cudaFree(devHistogramCopy);
		cudaFree(devHistogram);
		cudaFree(devInput);
		cudaFree(devOutput);
	}

	__global__ void FlyHistKernel(uint8_t* in, uint8_t* out, uint16_t histogramWidth, Coords* DiffRemY, Coords* DiffAddY, uint16_t diffLen, uint32_t frames, uint32_t height, uint32_t width, uint16_t iradius, float asqr, int threshold, bool moreIntense)
	{
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tempid >= frames * width)
			return;
		uint32_t z = tempid % frames;
		tempid /= frames;
		uint32_t x = tempid % width;

		if (x < iradius || z < iradius || x >= width - iradius || z >= frames - iradius)
			return;

		//auto CalculateDominance = moreIntense ? CalculateDominanceOverLessIntense :\\
			CalculateDominanceOverMoreIntense;

		uint8_t* histogram = new uint8_t[histogramWidth];

		for (int16_t k = -iradius; k <= iradius; k++)
			for (int16_t j = -iradius; j <= iradius; j++)
				for (int16_t i = -iradius; i <= iradius; i++)
					if (i * i + j * j + k * k <= asqr)
						histogram[in[((z + k) * height + iradius + j) * width + x + i]]++;

		for (uint32_t y = iradius; y < height - iradius; y++)
		{
			if (y != iradius)
			{
				for (uint32_t i = 0; i < diffLen; i++)      // compute by removing and adding delta pixels to histogram
				{
					histogram[in[CalculateIndex(DiffRemY[i], z, y, x, height, width)]]--;
					histogram[in[CalculateIndex(DiffAddY[i], z, y, x, height, width)]]++;
				}
			}

			uint16_t result = 0;
			uint16_t intensity = in[(z * height + y) * width + x] + threshold;

			if (!moreIntense)
				for (uint64_t i = 0; i < intensity; i++)
					result += histogram[i];
			else
				for (uint64_t i = intensity; i < histogramWidth; i++)
					result += histogram[i];
			out[(z * height + y) * width + x] = result;

			//uint16_t* result = new uint16_t;\
			out[(z * height + y) * width + x] = CalculateDominance(in[(z * height + y) * width + x] + threshold, \
				histogram, \
				diffLen, \
				histogramWidth * x, \
				result);
		}

		delete[] histogram;
	}


	void FlyingHistogram2(Image<uint8_t>& input, Image<uint8_t>& output, float radius, int threshold, bool moreIntense)
	{
		uint16_t iradius = std::ceil(radius);

		uint16_t DiffLen = 0, DiffLenZ = 0;
		Coords* DiffAddZ, * DiffRemZ, * DiffAddY, * DiffRemY;								 //array of coords of delta pixels

		DiffLenZ =	SetUpRadiusDifference(radius, &DiffAddZ, &DiffRemZ, true, Direction::Z); //number of delta pixels
		DiffLen  =	SetUpRadiusDifference(radius, &DiffAddY, &DiffRemY, true, Direction::Y);

		//to do anisotropic
		//float asqr = radius * radius;
		//float csqr = radiusZ * radiusZ;

		uint32_t frames = input.Frames(),
			height = input.Height(),
			width = input.Width();

		dim3 numBlocks((width * frames) / 1024 + 1, 1, 1);
		dim3 threadsPerBlock(1024, 1, 1);

		uint64_t size = input.GetSize();
		uint8_t* devInput, * devOutput;

		Coords* devDiffAddY, * devDiffRemY;
		auto tesd = sizeof(Coords);
		gpuErrchk(cudaMalloc(&devDiffAddY, DiffLen * sizeof(Coords)));
		gpuErrchk(cudaMalloc(&devDiffRemY, DiffLen * sizeof(Coords)));
		gpuErrchk(cudaMemcpy(devDiffAddY, DiffAddY, DiffLen * sizeof(Coords), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(devDiffRemY, DiffRemY, DiffLen * sizeof(Coords), cudaMemcpyHostToDevice));


		uint16_t histogramWidth = std::numeric_limits<uint8_t>::max() + 1;

		cudaMalloc(&devInput, size * sizeof(uint8_t));
		cudaMalloc(&devOutput, size * sizeof(uint8_t));

		gpuErrchk(cudaMemcpy(devInput, input.GetDataPtr(), size * sizeof(uint8_t), cudaMemcpyHostToDevice));

		FlyHistKernel<<<numBlocks, threadsPerBlock>>>
			(devInput, devOutput, histogramWidth, devDiffRemY, devDiffAddY, DiffLen, frames, height, width, iradius, radius * radius, threshold, moreIntense);

		cudaDeviceSynchronize();

		gpuErrchk(cudaMemcpy(output.GetDataPtr(), devOutput, size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

		cudaFree(devDiffAddY);
		cudaFree(devDiffRemY);

		cudaFree(devInput);
		cudaFree(devOutput);
	}

	__global__ void addKernel(int* c, const int* a, const int* b, int size) 
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < size) {
			c[i] = a[i] + b[i];
		}
	}

	void addWithCuda(int* c, const int* a, const int* b, int size) 
	{
		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;

		cudaMalloc((void**)&dev_c, size * sizeof(int));
		cudaMalloc((void**)&dev_a, size * sizeof(int));
		cudaMalloc((void**)&dev_b, size * sizeof(int));

		gpuErrchk(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

		//addKernel<<<2, (size + 1) / 2 >>> (dev_c, dev_a, dev_b, size);

		cudaDeviceSynchronize();

		gpuErrchk(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
	}
		
}