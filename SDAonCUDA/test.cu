#include <cuda_runtime.h>

#include "device_launch_parameters.h"
#include "SDAonCUDA.h"



namespace Test
{
	void asd(float a)
	{
		int b = a;
		int c = b * a;
		c++;
	}

	__global__ void SingleSDA(uint8_t* in, uint8_t* out, uint32_t frames, uint32_t height, uint32_t width, float radius, uint16_t iradius, int threshold, uint64_t size)
	{
		uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t x = id % width;
		uint32_t y = id / width % height;
		uint32_t z = id / width / height % frames;

		if (id / width / height / frames > 0)
			return;

		for (int16_t k = -iradius; k <= iradius; k++)
			if (0 <= z + k && z + k < frames)
				for (int16_t j = -iradius; j <= iradius; j++)
					if (0 <= y + j && y + j < height)
						for (int16_t i = -iradius; i <= iradius; i++)
							if (i * i + j * j + k * k <= radius * radius && 0 <= x + i && x + i < width)
								if (in[(z + k) * frames + (y + j) * height + (x + i) * width] >= in[z * frames + y * height + x * width] + threshold)
									out[z * frames + y * height + x * width]++;
	}

	//template<class InBitDepth, class OutBitDepth>
	//void GpuSDA(InBitDepth* image, OutBitDepth* output, float radius, int threshold)
	void GpuSDA(uint8_t* input, uint8_t* output, float radius, int threshold, uint32_t frames, uint32_t height, uint32_t width)
	{
		uint64_t size = frames * height * width;
		uint8_t* gInput,* gOutput;

		cudaMalloc(&gInput,  size * sizeof(uint8_t));
		cudaMalloc(&gOutput, size * sizeof(uint8_t));

		cudaMemcpy(gInput, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

		dim3 grid_size(2500);
		dim3 block_size(size/2500);

		//std::cout << "start\n";
		uint16_t iradius = radius + 0.999;
		SingleSDA<<<grid_size, block_size>>>(input, output, frames, height, width, radius, iradius, threshold, size);

		//std::cout << "end\n";
		cudaMemcpy(output, gOutput, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	}
		
}