#include <cuda_runtime.h>

#include "device_launch_parameters.h"
#include "SDAonCUDA.h"


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
	__global__ void gpuSDAmultiDim(uint8_t* in, uint8_t* out, uint32_t frames, uint32_t height, uint32_t width, float radius, uint16_t iradius, int threshold, uint64_t size)
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
							if (i * i + j * j + k * k <= radius * radius && 0 <= x + i && x + i < width)
								if (in[((z + k) * height + y + j) * width + x + i] >= in[(z * height + y) * width + x] + threshold)
									out[(z * height + y) * width + x]++;
	}
	__global__ void gpuSDAsingleDim(uint8_t* in, uint8_t* out, uint32_t frames, uint32_t height, uint32_t width, float radius, uint16_t iradius, int threshold, uint64_t size)
	{
		uint64_t tempid = threadIdx.x + blockIdx.x * blockDim.x;
 		uint32_t x = (tempid) % width;
		tempid /= width;
		uint32_t y = tempid % height;
		tempid /= height;
		uint32_t z = tempid % frames;

		if (tempid / frames > 0)
			return;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (i * i + j * j + k * k <= radius * radius && 0 <= x + i && x + i < width)
								if (in[((z + k) * height + y + j) * width + x + i] >= in[(z * height + y) * width + x] + threshold)
									out[(z * height + y) * width + x]++;
	}

	//template<class InBitDepth, class OutBitDepth>
	//void GpuSDA(InBitDepth* image, OutBitDepth* output, float radius, int threshold)
	void SDA(uint8_t* input, uint8_t* output, float radius, int threshold, uint32_t frames, uint32_t height, uint32_t width)
	{
		//cudaDeviceProp prop;
		//cudaGetDeviceProperties(&prop, 0);
		//std::cout << "\ngrid:" << prop.maxGridSize[0] << "\n" << prop.maxGridSize[1] << "\n" << prop.maxGridSize[2] << "\n" << prop.maxSurface3D[0];
		uint64_t size = frames * height * width;
		uint8_t* dev_Input,* dev_Output;

		cudaMalloc((void**)&dev_Input,  size * sizeof(uint8_t));
		cudaMalloc((void**)&dev_Output, size * sizeof(uint8_t));

		cudaMemcpy(dev_Input, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

		uint16_t iradius = (uint16_t)radius + 0.999;

		//dim3 numBlocks(64, 8, 8);
		//dim3 threadsPerBlock(8, 8, 8);
		//SingleSDA<<<numBlocks, threadsPerBlock>>>(dev_Input, dev_Output, frames, height, width, radius, iradius, threshold, size);
		
		dim3 numBlocks(size / 1024 + 1, 1, 1);
		dim3 threadsPerBlock(1024, 1, 1);
		gpuSDAsingleDim<<<numBlocks, threadsPerBlock>>>(dev_Input, dev_Output, frames, height, width, radius, iradius, threshold, size);

		cudaDeviceSynchronize();

		cudaMemcpy(output, dev_Output, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

		cudaFree(dev_Input);
		cudaFree(dev_Output);
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

		addKernel <<<2, (size + 1) / 2 >>> (dev_c, dev_a, dev_b, size);

		cudaDeviceSynchronize();

		cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
	}
		
}