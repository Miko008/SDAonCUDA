#pragma once

namespace GPU
{
	/// <summary>
	/// Standard 3D SDA parallel function.
	/// </summary>
	/// <param name="input">image</param>
	/// <param name="output">image</param>
	/// <param name="radius">of calculated region</param>
	/// <param name="threshold"></param>
	void SDAExt(Image<uint8_t>& input, Image<uint8_t>& output, float radius, int threshold, bool moreIntense);
	void SDAExt(Image<uint8_t>& input, Image<uint16_t>& output, float radius, int threshold, bool moreIntense);
	void SDAExt(Image<uint8_t>& input, Image<uint32_t>& output, float radius, int threshold, bool moreIntense);

	/// <summary>
	/// Flying Histogram calculating by each y column
	/// </summary>
	/// <param name="input">image</param>
	/// <param name="output">image</param>
	/// <param name="radius">of calculated region</param>
	/// <param name="threshold"></param>
	/// <param name="moreIntense">: true  - count lighter pixels, false - count darker pixels</param>
	void FlyingHistogramExt(Image<uint8_t>& input, Image<uint8_t>& output, float radius, int threshold, bool moreIntense);
	void FlyingHistogramExt(Image<uint8_t>& input, Image<uint16_t>& output, float radius, int threshold, bool moreIntense);
	void FlyingHistogramExt(Image<uint8_t>& input, Image<uint32_t>& output, float radius, int threshold, bool moreIntense);
}