#pragma once

namespace GPU
{
	//template<class InBitDepth, class OutBitDepth>
	void SDA(Image<uint8_t> &input, Image<uint8_t> &output, float radius, int threshold);

	void FlyingHistogram(Image<uint8_t> &input, Image<uint8_t> &output, float radius, int threshold, bool moreIntense);
	
	void FlyingHistogram2(Image<uint8_t> &input, Image<uint8_t> &output, float radius, int threshold, bool moreIntense);

	void addWithCuda(int* c, const int* a, const int* b, int size);
}