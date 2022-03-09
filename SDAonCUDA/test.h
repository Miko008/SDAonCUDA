#pragma once

namespace Test
{
	void asd(float a);

	//template<class InBitDepth, class OutBitDepth>
	//void GpuSDA(InBitDepth* image, OutBitDepth* output, float radius, int threshold);
	
	void GpuSDA(uint8_t* image, uint8_t* output, float radius, int threshold,
		uint32_t frames, uint32_t height, uint32_t width);
}