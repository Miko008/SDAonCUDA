#include <cuda_runtime.h>
#include <cmath>
#include <limits>

#include "device_launch_parameters.h"
#include "main.h"
#include "cuda.h"

constexpr uint32_t THREADS_PER_BLOCK = 1024;

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
#pragma region Device functions
	//GPU functions callable only from GPU

	__device__ bool MoreIntense(uint8_t lhs, uint8_t rhs, int thresh)
	{
		if (rhs + thresh < 0) //to do - ensure no wraping on device || std::numeric_limits<uint8_t>::max() - rhs < thresh)
			return false;
		return lhs >= rhs + thresh;
	}

	__device__ bool LessIntense(uint8_t lhs, uint8_t rhs, int thresh)
	{
		if (rhs + thresh < 0)
			return false;
		return lhs <= rhs + thresh;
	}

	/// <summary>
	/// Calculates darker pixels from histogram
	/// </summary>
	/// <param name="intensity"></param>
	/// <param name="histogram"></param>
	/// <param name="diffLen">length</param>
	/// <returns>pixel count</returns>
	__device__ uint8_t CalculateDominanceOverMoreIntense(int intensity, uint8_t* histogram, uint16_t diffLen)
	{
		uint16_t result = 0;
		uint32_t end = intensity > diffLen ? diffLen : intensity;    //select end value not higher than diffLen

		for (uint32_t i = 0; i < end; i++)    //add numbers of pixels that are <= pixel + threshold
			result += histogram[i];

		return result;
	}

	/// <summary>
	/// Calculates lighter pixels from histogram
	/// </summary>
	/// <param name="intensity"></param>
	/// <param name="histogram"></param>
	/// <param name="diffLen">length</param>
	/// <returns>pixel count</returns>
	__device__ uint8_t CalculateDominanceOverLessIntense(int intensity, uint8_t* histogram, uint16_t diffLen)
	{
		uint16_t result = 0;
		uint32_t start = intensity > 0 ? intensity : 0;		//select end value not lower than 0 (threshold can is int, could be negative)

		for (uint32_t i = start; i < diffLen; i++)			//add numbers of pixels that are >= pixel + threshold
			result += histogram[i];

		return result;
	}

	/// <summary>
	/// Calculates index of delta pixel using Coord reference
	/// </summary>
	/// <param name="Diff">Coord reference</param>
	/// <param name="z">of calculated pixel</param>
	/// <param name="y">of calculated pixel</param>
	/// <param name="x">of calculated pixel</param>
	/// <param name="height">of image</param>
	/// <param name="width">of image</param>
	/// <returns>delta pixel index</returns>
	__device__ uint32_t CalculateIndex(Coords& Diff, uint32_t z, uint32_t y, uint32_t x, uint32_t height, uint32_t width)
	{
		return ((z + Diff.z) * height + y + Diff.y) * width + x + Diff.x;
	}

	__device__ uint8_t SingleSDA(uint8_t* in, uint32_t frames, uint32_t height, uint32_t width, uint32_t x, uint32_t y, uint32_t z, float asqr, uint16_t iradius, int threshold, bool moreIntense)
	{
		uint8_t result = 0;
		auto condition = moreIntense ? MoreIntense : LessIntense;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (0 <= x + i && x + i < width && i * i + j * j + k * k <= asqr)
								if (condition(in[((z + k) * height + y + j) * width + x + i], in[(z * height + y) * width + x], threshold))
									result++;
	}

#pragma endregion


#pragma region Global functions
	//GPU functions callable from CPU

	/// <summary>
	/// SDA parallel function callable by 3 dimension blocks
	/// </summary>
	/// <param name="in">input image</param>
	/// <param name="out">output image</param>
	/// <param name="frames">dimension</param>
	/// <param name="height">dimension</param>
	/// <param name="width">dimension</param>
	/// <param name="rsqr">radius squared</param>
	/// <param name="iradius">ceiled radius</param>
	/// <param name="threshold"></param>
	/// <param name="size">image size</param>
	template <class T>
	__global__ void SDAKernel3D(uint8_t* in, T* out, uint32_t frames, uint32_t height, uint32_t width, float asqr, uint16_t iradius, int threshold, uint64_t size, bool moreIntense)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
		uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;

		if (x >= width || y >= height || z >= frames)
			return;

		auto condition = moreIntense ? MoreIntense : LessIntense;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (0 <= x + i && x + i < width && i * i + j * j + k * k <= asqr)
								if (condition(in[((z + k) * height + y + j) * width + x + i], in[(z * height + y) * width + x], threshold))
									out[(z * height + y) * width + x]++;
	}

	/// <summary>
	/// SDA parallel function callable by 1 dimension
	/// </summary>
	/// <param name="in">input image</param>
	/// <param name="out">output image</param>
	/// <param name="frames">dimension</param>
	/// <param name="height">dimension</param>
	/// <param name="width">dimension</param>
	/// <param name="rsqr">radius squared</param>
	/// <param name="iradius">ceiled radius</param>
	/// <param name="threshold"></param>
	/// <param name="size">image size</param>
	template <class T>
	__global__ void SDAKernel1D(uint8_t* in, T* out, uint32_t frames, uint32_t height, uint32_t width, float asqr, uint16_t iradius, int threshold, uint64_t size, bool moreIntense)
	{
		//todo omit using division operations
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tempid >= size)
			return;
		uint32_t x = tempid % width;
		tempid /= width;
		uint32_t y = tempid % height;
		tempid /= height;
		uint32_t z = tempid % frames;

		auto condition = moreIntense ? MoreIntense : LessIntense;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (0 <= x + i && x + i < width && i * i + j * j + k * k <= asqr)
								if (condition(in[((z + k) * height + y + j) * width + x + i], in[(z * height + y) * width + x], threshold))
									out[(z * height + y) * width + x]++;
	}

	/// <summary>
	/// @deprecated
	/// First histogram calculation
	/// </summary>
	/// <param name="in">input image</param>
	/// <param name="histogram">output histogram</param>
	/// <param name="histogramWidth">single histogram size</param>
	/// <param name="frames">dimension</param>
	/// <param name="height">dimension</param>
	/// <param name="width">dimension</param>
	/// <param name="radius"></param>
	/// <param name="iradius">ceiled radius</param>
	/// <param name="threshold"></param>
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

	/// <summary>
	/// @deprecated
	/// Flying Histogram calculation parallel in x dimension
	/// </summary>
	/// <param name="in">input image</param>
	/// <param name="out">output image</param>
	/// <param name="histogram">calculated histogram</param>
	/// <param name="histogramCopy">reserved array for histogram copy</param>
	/// <param name="histogramWidth">single histogram size</param>
	/// <param name="DiffRemZ">Coord array of deltapixels to remove, when moving in Z direction</param>
	/// <param name="DiffAddZ">Coord array of deltapixels to add, when moving in Z direction</param>
	/// <param name="DiffRemY">Coord array of deltapixels to remove, when moving in Y direction</param>
	/// <param name="DiffAddY">Coord array of deltapixels to add, when moving in Y direction</param>
	/// <param name="diffLen">size of Coord arrays (should be same value for all arrays)</param>
	/// <param name="frames">dimension of input/output image</param>
	/// <param name="height">dimension of input/output image</param>
	/// <param name="width" >dimension of input/output image</param>
	/// <param name="iradius">ceiled radius</param>
	/// <param name="threshold"></param>
	/// <param name="moreIntense">: true  - count lighter pixels, false - count darker pixels </param>
	__global__ void FHKernel(uint8_t* in, uint8_t* out, uint8_t* histogram, uint8_t* histogramCopy, uint16_t histogramWidth, Coords* DiffRemZ, Coords* DiffAddZ, Coords* DiffRemY, Coords* DiffAddY, uint16_t diffLen, uint32_t frames, uint32_t height, uint32_t width, uint16_t iradius, int threshold, bool moreIntense)
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

	/// <summary>
	/// Better version of parallel (for each y column) Flying Histogram, using histogram in single kernel stack instead of shared memory
	/// </summary>
	/// <param name="in">input image</param>
	/// <param name="out">output image</param>
	/// <param name="histogramWidth">single histogram size</param>
	/// <param name="DiffRemY">Coord array of deltapixels to remove, when moving in Y direction</param>
	/// <param name="DiffAddY">Coord array of deltapixels to add, when moving in Y direction</param>
	/// <param name="diffLen">size of Coord arrays (should be same value for all arrays)</param>
	/// <param name="frames">dimension of input/output image</param>
	/// <param name="height">dimension of input/output image</param>
	/// <param name="width" >dimension of input/output image</param>
	/// <param name="iradius">ceiled radius</param>
	/// <param name="asqr">radius squared</param>
	/// <param name="threshold"></param>
	/// <param name="moreIntense">: true  - count lighter pixels, false - count darker pixels </param>
	template <class T>
	__global__ void FlyHistKernel(uint8_t* in, T* out, uint16_t histogramWidth, Coords* DiffRemY, Coords* DiffAddY, uint16_t diffLen, uint32_t frames, uint32_t height, uint32_t width, uint16_t iradius, float asqr, int threshold, bool moreIntense)
	{
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tempid >= frames * width)
			return;
		uint32_t z = tempid % frames;
		tempid /= frames;
		uint32_t x = tempid % width;
		//^caclulate x and z based on thread id

		if (x < iradius || z < iradius || x >= width - iradius || z >= frames - iradius)
			return;			//return when out of bonds

		//auto CalculateDominance = moreIntense ? CalculateDominanceOverLessIntense :\
			CalculateDominanceOverMoreIntense;
		//^set function ptr to selected dominance calculation

		T histogram[256] = { 0 };	//new uint8_t[histogramWidth];

		for (int16_t k = -iradius; k <= iradius; k++)
			for (int16_t j = -iradius; j <= iradius; j++)
				for (int16_t i = -iradius; i <= iradius; i++)
					if (i * i + j * j + k * k <= asqr)
						histogram[in[((z + k) * height + iradius + j) * width + x + i]]++;
		//^calculate first histogram

		for (uint32_t y = iradius; y < height - iradius; y++)		//move histogram on y dim
		{
			if (y != iradius)
			{
				for (uint32_t i = 0; i < diffLen; i++)      // compute by removing and adding delta pixels to histogram
				{
					histogram[in[CalculateIndex(DiffRemY[i], z, y, x, height, width)]]--;
					histogram[in[CalculateIndex(DiffAddY[i], z, y, x, height, width)]]++;
				}
			}

			//cacluate dominance in scope (temporary)
			T result = 0;
			int zeroTest = in[(z * height + y) * width + x] + threshold;
			uint16_t intensity = zeroTest > 0 ? zeroTest : 0;

			if (!moreIntense)
				for (uint64_t i = 0; i <= intensity; i++)
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

		//delete[] histogram;
	}

	template <class T>
	__global__ void MarginSDA(uint8_t* in, T* out, uint16_t histogramWidth, uint32_t frames, uint32_t height, uint32_t width, uint16_t iradius, float asqr, int threshold, bool moreIntense)
	{		
		//todo omit using division operations
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tempid >= frames * height * width)
			return;
		uint32_t x = tempid % width;
		tempid /= width;
		uint32_t y = tempid % height;
		tempid /= height;
		uint32_t z = tempid % frames;
		//^caclulate x, y and z based on thread id

		if ((x >= iradius && x < width - iradius) && (y >= iradius && y < height - iradius) && (z >= iradius && z < frames - iradius))
			return;
		//^skip core calculated by FH

		T result = 0;
		auto condition = moreIntense ? MoreIntense : LessIntense;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (0 <= x + i && x + i < width && i * i + j * j + k * k <= asqr)
								if (condition(in[((z + k) * height + y + j) * width + x + i], in[(z * height + y) * width + x], threshold))
									result++;

		out[(z * height + y) * width + x] = result; //SingleSDA(in, frames, height, width, x, y, z, asqr, iradius, threshold, moreIntense);
	}

	template <class T>
	__global__ void MarginSDA3D(uint8_t* in, T* out, uint16_t histogramWidth, uint32_t frames, uint32_t height, uint32_t width, uint16_t iradius, float asqr, int threshold, bool moreIntense)
	{
		uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
		uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
		//^caclulate x, y and z based on thread id

		if (x >= width || y >= height || z >= frames)
			return;
		//^skip when out of bonds

		if ((x >= iradius && x < width - iradius) && (y >= iradius && y < height - iradius) && (z >= iradius && z < frames - iradius))
			return;
		//^skip core calculated by FH

		T result = 0;
		auto condition = moreIntense ? MoreIntense : LessIntense;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (0 <= x + i && x + i < width && i * i + j * j + k * k <= asqr)
								if (condition(in[((z + k) * height + y + j) * width + x + i], in[(z * height + y) * width + x], threshold))
									result++;

		out[(z * height + y) * width + x] = result;
	}

	__global__ void addKernel(int* c, const int* a, const int* b, int size)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < size) {
			c[i] = a[i] + b[i];
		}
	}

#pragma endregion

#pragma region CPU functions

	template <class T>
	void SDA(Image<uint8_t>& input, Image<T>& output, float radius, int threshold, bool moreIntense)
	{
		//Todo fix arg templates - linker error
		//template<class InBitDepth, class OutBitDepth>
		uint8_t* devInput;
		T* devOutput;
		uint64_t size = input.GetSize();
		cudaMalloc((void**)&devInput, size * sizeof(uint8_t));
		cudaMalloc((void**)&devOutput, size * sizeof(T));

		cudaMemcpy(devInput, input.GetDataPtr(), size * sizeof(uint8_t), cudaMemcpyHostToDevice);

		uint16_t iradius = std::ceil(radius);

		dim3 numBlocks(input.Width() / 8 + 1, input.Height() / 8 + 1, input.Frames() / 8 + 1);
		dim3 threadsPerBlock(8, 8, 8);
		SDAKernel3D<<<numBlocks, threadsPerBlock>>>(devInput, devOutput, input.Frames(), input.Height(), input.Width(), radius * radius, iradius, threshold, size, moreIntense);

		//dim3 numBlocks(size / THREADS_PER_BLOCK + 1, 1, 1);
		//dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
		//SDAKernel1D <<<numBlocks, threadsPerBlock>>>
		//	(devInput, devOutput, input.Frames(), input.Height(), input.Width(), radius * radius, iradius, threshold, size, moreIntense);

		cudaDeviceSynchronize();

		cudaMemcpy(output.GetDataPtr(), devOutput, size * sizeof(T), cudaMemcpyDeviceToHost);

		cudaFree(devInput);
		cudaFree(devOutput);
	}

	template <class T>
	void FlyingHistogram(Image<uint8_t>& input, Image<T>& output, float radius, int threshold, bool moreIntense)
	{
		uint16_t iradius = std::ceil(radius);

		uint16_t DiffLen = 0, DiffLenZ = 0;
		Coords* DiffAddZ, * DiffRemZ, * DiffAddY, * DiffRemY;								 //array of coords of delta pixels

		DiffLenZ = SetUpRadiusDifference(radius, &DiffAddZ, &DiffRemZ, true, Direction::Z); //number of delta pixels
		DiffLen  = SetUpRadiusDifference(radius, &DiffAddY, &DiffRemY, true, Direction::Y);

		//to do anisotropic
		//float asqr = radius * radius;
		//float csqr = radiusZ * radiusZ;

		uint32_t frames = input.Frames(),
				 height = input.Height(),
				 width  = input.Width();
		uint64_t size   = input.GetSize();

		dim3 numBlocks((width * frames) / THREADS_PER_BLOCK + 1, 1, 1);
		dim3 numBlocksMargin(width / 8 + 1, height / 8 + 1, frames / 8 + 1);
		dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
		dim3 threadsPerBlockMargin(8, 8, 8);

		uint8_t* devInput;
		T* devOutput;

		Coords* devDiffAddY, * devDiffRemY;
		gpuErrchk(cudaMalloc(&devDiffAddY, DiffLen * sizeof(Coords)));
		gpuErrchk(cudaMalloc(&devDiffRemY, DiffLen * sizeof(Coords)));
		gpuErrchk(cudaMemcpy(devDiffAddY, DiffAddY, DiffLen * sizeof(Coords), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(devDiffRemY, DiffRemY, DiffLen * sizeof(Coords), cudaMemcpyHostToDevice));


		uint16_t histogramWidth = std::numeric_limits<uint8_t>::max() + 1;

		cudaMalloc(&devInput, size * sizeof(uint8_t));
		cudaMalloc(&devOutput, size * sizeof(T));

		gpuErrchk(cudaMemcpy(devInput, input.GetDataPtr(), size * sizeof(uint8_t), cudaMemcpyHostToDevice));

		//calculate inside of image with FH
		FlyHistKernel <<<numBlocks, threadsPerBlock >>>
			(devInput, devOutput, histogramWidth, devDiffRemY, devDiffAddY, DiffLen, frames, height, width, iradius, radius * radius, threshold, moreIntense);

		MarginSDA3D <<<numBlocksMargin, threadsPerBlockMargin >>>
			(devInput, devOutput, histogramWidth, frames, height, width, iradius, radius * radius, threshold, moreIntense);

		cudaDeviceSynchronize();

		gpuErrchk(cudaMemcpy(output.GetDataPtr(), devOutput, size * sizeof(T), cudaMemcpyDeviceToHost));

		cudaFree(devDiffAddY);
		cudaFree(devDiffRemY);

		cudaFree(devInput);
		cudaFree(devOutput);
	}

	void SDAExt(Image<uint8_t>& input, Image<uint8_t>& output, float radius, int threshold, bool moreIntense) {
		SDA(input, output, radius, threshold, moreIntense);
	}
	void SDAExt(Image<uint8_t>& input, Image<uint16_t>& output, float radius, int threshold, bool moreIntense) {
		SDA(input, output, radius, threshold, moreIntense);
	}
	void SDAExt(Image<uint8_t>& input, Image<uint32_t>& output, float radius, int threshold, bool moreIntense) {
		SDA(input, output, radius, threshold, moreIntense);
	}

	void FlyingHistogramExt(Image<uint8_t>& input, Image<uint8_t>& output, float radius, int threshold, bool moreIntense) {
		FlyingHistogram(input, output, radius, threshold, moreIntense);
	}
	void FlyingHistogramExt(Image<uint8_t>& input, Image<uint16_t>& output, float radius, int threshold, bool moreIntense) {
		FlyingHistogram(input, output, radius, threshold, moreIntense);
	}
	void FlyingHistogramExt(Image<uint8_t>& input, Image<uint32_t>& output, float radius, int threshold, bool moreIntense) {
		FlyingHistogram(input, output, radius, threshold, moreIntense);
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

#pragma endregion
}