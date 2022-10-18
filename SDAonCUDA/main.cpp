// SDAonCUDA.cpp : Defines the entry point for the application.
//

#include <cstring>
#include <chrono>
#include <cmath>

#include "main.h"
#include "tinytiffreader.h"
#include "tinytiffwriter.h"
#include "cuda.h"
#include "options.hpp"
#include "logger.hpp"

using namespace std::string_literals;

#define SWEET_LOGGER_RELEASE

#pragma region Image class

template<class BitDepth>
Image<BitDepth>::Image()
{
    width = 0;
    height = 0;
    frames = 0;
    data = nullptr;
}

template<class BitDepth>
Image<BitDepth>::Image(const Image& pattern)
{
    width  = pattern.width;
    height = pattern.height;
    frames = pattern.frames;
    data   = new BitDepth[GetSize()];
    memcpy(data, pattern.data, GetSize() * sizeof(BitDepth));
}

template<class BitDepth>
Image<BitDepth>::Image(const Image& pattern, BitDepth val)
{
    width = pattern.width;
    height = pattern.height;
    frames = pattern.frames;
    data = new BitDepth[GetSize()];
    memset(data, val, GetSize() * sizeof(BitDepth));
}

template<class BitDepth>
Image<BitDepth>::Image(uint32_t _width, uint32_t _height, uint32_t _frames)
{
    width = _width;
    height = _height;
    frames = _frames;
    data = new BitDepth[GetSize()];
    memset(data, 0, GetSize() * sizeof(BitDepth));
}


template<class BitDepth>
Image<BitDepth>::~Image()
{
    delete[] data;
}

template<class BitDepth>
Image<BitDepth>& Image<BitDepth>::operator=(const Image<BitDepth>& pattern)
{
    return Image<BitDepth>(pattern);
}

template<class BitDepth>
BitDepth& Image<BitDepth>::Image::operator()(uint32_t z, uint32_t y, uint32_t x)
{
    return data[(z * static_cast<uint64_t>(height) + y) * width + x];
}


template<class BitDepth>
uint64_t Image<BitDepth>::Index(uint32_t z, uint32_t y, uint32_t x) const
{
    return (z * static_cast<uint64_t>(height) + y) * width + x;
}


template<class BitDepth>
uint64_t Image<BitDepth>::GetSize() const
{
    return frames * static_cast<uint64_t>(height) * width;
}


template<class BitDepth>
void Image<BitDepth>::SetSize(uint32_t _width, uint32_t _height, uint32_t _frames)
{
    if (data != nullptr)
        delete[] data;
    width = _width;
    height = _height;
    frames = _frames;
    data = new BitDepth[GetSize()];
}


template<class BitDepth>
bool Image<BitDepth>::SetSlide(uint32_t frame, BitDepth* newslide)
{
    if (frame < frames)
    {
        BitDepth* slidestart = (data + frame * static_cast<uint64_t>(width) * height);
        memcpy(slidestart, newslide, width * static_cast<uint64_t>(height) * sizeof(BitDepth));
    }
    else
        return false;
    return true;
}

template<class BitDepth>
BitDepth Image<BitDepth>::MaxValue() const
{
    BitDepth max = data[0];
    for (BitDepth* p = data; p < data + GetSize(); ++p)
        if (max < *p)
            max = *p;
    return max;
}


template<class BitDepth>
void Image<BitDepth>::Normalize()
{
    BitDepth max = MaxValue();
    if (max == 0)
        return;     //skip if empty
    size_t newMax = (size_t)(std::numeric_limits<BitDepth>::max());
    for (BitDepth* p = data; p < data + GetSize(); ++p)
        *p = (*p * newMax) / max;
}

template<class BitDepth>
void Image<BitDepth>::Normalize(size_t newMax)
{
    BitDepth max = MaxValue();
    if (max == 0)
        return;     //skip if empty
    for (BitDepth* p = data; p < data + GetSize(); ++p)
        *p = (*p * newMax) / max;
}

template<class BitDepth>
void Image<BitDepth>::Clear()
{
    memset(data, 0, GetSize() * sizeof(BitDepth));
}


template<class BitDepth>
bool Image<BitDepth>::Image::operator==(Image& rhs) const
{
    return frames == rhs.frames && height == rhs.height && width == rhs.width && !memcmp(data, rhs.data, GetSize());
}
#pragma endregion


#pragma region TiffIO


template<class BitDepth>
bool ReadTiff(Image<BitDepth>& image, const char* filename)
{
    bool ok = false;
    LOG("\nReading '" + std::string(filename) + "'\n");
    TinyTIFFReaderFile* tiffr = TinyTIFFReader_open(filename);
    if (!tiffr) {
        LOG("\nERROR reading (not existent, not accessible or no TIFF file)\n");
        return false;
    }
    else {
        if (TinyTIFFReader_wasError(tiffr)) LOG("\nERROR:"s + TinyTIFFReader_getLastError(tiffr) + "\n"s);

        uint32_t width = TinyTIFFReader_getWidth(tiffr);
        uint32_t height = TinyTIFFReader_getHeight(tiffr);
        uint32_t frames = TinyTIFFReader_countFrames(tiffr);
        BitDepth* slide = new BitDepth[width * static_cast<uint64_t>(height)];
        image.SetSize(width, height, frames);

        if (TinyTIFFReader_wasError(tiffr)) LOG("\nERROR:"s + TinyTIFFReader_getLastError(tiffr) + "\n"s);
        else ok = true;

        for (uint32_t frame = 0; ok; frame++)
        {
            TinyTIFFReader_getSampleData(tiffr, slide, 0);
            image.SetSlide(frame, slide);

            if (TinyTIFFReader_wasError(tiffr))
            {
                ok = false;
                LOG("\nERROR:"s + TinyTIFFReader_getLastError(tiffr) + "\n"s);
            }
            if (!TinyTIFFReader_readNext(tiffr))
                break;
        }
        delete[] slide;
        LOG("\nread and checked all frames: "s + ((ok) ? "SUCCESS"s : "ERROR"s) + " \n"s);
    }
    TinyTIFFReader_close(tiffr);
    if (!ok)
        return false;
    return true;
}

template<class BitDepth>
bool SaveTiff(Image<BitDepth>& image, const char* filename)
{
    TinyTIFFWriterFile* tiff = TinyTIFFWriter_open(filename, sizeof(BitDepth) * 8, TinyTIFFWriter_UInt, 1, image.Width(), image.Height(), TinyTIFFWriter_Greyscale);
    //bits per sample constant cause of some errors

    LOG("\nSaving as '"s + filename + "'\n"s);
    if (tiff)
    {
        for (size_t f = 0; f < image.Frames(); f++)
        {
            int res = TinyTIFFWriter_writeImage(tiff, image.GetDataPtr() + (f * image.Width() * image.Height())); //TinyTIFF_Planar   TinyTIFF_Chunky
            if (res != TINYTIFF_TRUE)
            {
                LOG("\nERROR: error writing image data into '"s + filename + "'! MESSAGE: "s + TinyTIFFWriter_getLastError(tiff) + "\n"s);
                TinyTIFFWriter_close(tiff);
                return false;
            }
        }
        TinyTIFFWriter_close(tiff);
        LOG("\nFile saved as '"s + filename + "'\n"s);
        return true;
    }
    LOG("\nERROR: could not open '"s + filename + "' for writing!\n"s);
    return false;
}

template<class BitDepth>
bool CropTiff(Image<BitDepth>& image, Image<BitDepth>& croppedImage,
    uint32_t width0, uint32_t height0, uint32_t frames0,
    uint32_t width1, uint32_t height1, uint32_t frames1)
{
    if (width1 < width0 || height1 < height0 || frames1 < frames0)
        return false;
    if (image.Width() < width1 || image.Height() < height1 || image.Frames() < frames1)
        return false;
    croppedImage.SetSize(width1 - width0, height1 - height0, frames1 - frames0);

    uint32_t width  = image.Width(),
             height = image.Height(),
             frames = image.Frames(),
             newWidth  = croppedImage.Width(),
             newHeight = croppedImage.Height(),
             newFrames = croppedImage.Frames();

    for (uint32_t z = 0; z < newFrames; z++)
        for (uint32_t y = 0; y < newHeight; y++)
            for (uint32_t x = 0; x < newWidth; x++)
                croppedImage(z, y, x) = image(frames0 + z, height0 + y, width0 + x);

    return true;
}

#pragma endregion
template<class InBitDepth>
uint8_t SymmetricalDominance(Image<InBitDepth>& image, int threshold, uint32_t z, uint32_t y, uint32_t x, uint16_t k, uint16_t i, uint16_t j)
{
    uint8_t result = 0;
    int value = image(z, y, j) + threshold;
    if (image(z + k, y + j, x + i) >= value)
        result++;
    if (image(z + k, y + j, x - i) >= value)
        result++;
    if (image(z + k, y - j, x + i) >= value)
        result++;
    if (image(z + k, y - j, x - i) >= value)
        result++;
    if (image(z - k, y + j, x + i) >= value)
        result++;
    if (image(z - k, y + j, x - i) >= value)
        result++;
    if (image(z - k, y - j, x + i) >= value)
        result++;
    if (image(z - k, y - j, x - i) >= value)
        result++;
    return result;
}

template<class InBitDepth, class OutBitDepth>
void SDA(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold)
{
    uint32_t width  = image.Width(),
             height = image.Height(),
             frames = image.Frames();
    uint16_t iradius = std::ceil(radius);

    for (uint32_t z = iradius; z < frames - iradius; z++)
    {
        std::cout << z << " ";
        for (uint32_t y = iradius; y < height - iradius; y++)
            for (uint32_t x = iradius; x < width - iradius; x++)
                for (int16_t k = 0; k <= iradius; k++)
                    for (int16_t j = 0; j <= iradius; j++)
                        for (int16_t i = 0; i <= iradius; i++)
                            if (i * i + j * j + k * k <= radius * radius)
                                output(z, y, x) += SymmetricalDominance(image, threshold, z, y, x, k, i, j);
    }
}

template<class InBitDepth, class OutBitDepth>
void SDAborderless(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold)
{
    uint32_t width  = image.Width(),
             height = image.Height(),
             frames = image.Frames();
    uint16_t iradius = std::ceil(radius);

    for (uint32_t z = 0; z < frames; z++)
    {
        std::cout << z << " ";
        for (uint32_t y = 0; y < height; y++)
            for (uint32_t x = 0; x < width; x++)
                for (int16_t k = -iradius; k <= iradius; k++)
                    if (0 <= z + k && z + k < frames)
                        for (int16_t j = -iradius; j <= iradius; j++)
                            if (0 <= y + j && y + j < height)
                                for (int16_t i = -iradius; i <= iradius; i++)
                                    if (i * i + j * j + k * k <= radius * radius && 0 <= x + i && x + i < width)
                                        if (image(z + k, y + j, x + i) >= image(z, y, x) + threshold)
                                            output(z, y, x)++;
    }
}

template<class InBitDepth, class OutBitDepth>
void SDAborderless2D(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold)
{
    uint32_t width  = image.Width(),
             height = image.Height();
    uint16_t iradius = std::ceil(radius);

    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            for (int16_t j = -iradius; j <= iradius; j++)
                if (0 <= y + j && y + j < height)
                    for (int16_t i = -iradius; i <= iradius; i++)
                        if (i * i + j * j <= radius * radius && 0 <= x + i && x + i < width)
                            if (image(0, y + j, x + i) >= image(0, y, x) + threshold)
                                output(0, y, x)++;
}

bool sphereCondition(int16_t i, int16_t j, int16_t k, float asqr, float csqr)
{
    return (i * i + j * j + k * k <= asqr);
}

bool anisotropicCondition(int16_t i, int16_t j, int16_t k, float asqr, float csqr)
{
    return (asqr * k * k + csqr * (i * i + j * j) <= asqr * csqr);
}

uint16_t SetUpRadiusDifference(float radius, Coords** DifferenceAddPtr, Coords** DifferenceRemPtr, bool threeDim, Direction dir)
{
    return SetUpRadiusDifference(radius, radius, DifferenceAddPtr, DifferenceRemPtr, threeDim, false, dir);
}

uint16_t SetUpRadiusDifference(float radius, float radiusZ, Coords** DifferenceAddPtr, Coords** DifferenceRemPtr, bool threeDim, bool anisotropic, Direction dir)
{
    if (!threeDim && anisotropic)                   
        return 0;

    auto histogramCondition = anisotropic ? anisotropicCondition : sphereCondition;

    uint16_t iradius = std::ceil(radiusZ > radius ? radiusZ : radius);  //choose larger radius
    uint16_t margin = iradius * 2 + 2;              //to fit 2 offset spheres
    uint16_t numberOfDifs = 0;                      //number of delta pixels

    uint16_t z = iradius, 
             y = iradius, 
             x = iradius;

    Image<uint8_t> tempArray = threeDim ? Image<uint8_t>(margin, margin, margin) : Image<uint8_t>(margin, margin, 1);

    if (!threeDim)
        z = 0;

    float asqr = radius * radius;
    float csqr = radiusZ * radiusZ;

    for (uint8_t origin = 0; origin < 2; origin++)  //origin offset + id to mark 2 offset spheres
    {
        switch (dir)
        {
        default:
        case Direction::Z:
            z = iradius + origin;
            break;
        case Direction::Y:
            y = iradius + origin;
            break;
        case Direction::X:
            x = iradius + origin;
            break;
        }
        numberOfDifs = 0;
        if (threeDim)
        {
            for (int16_t k = -iradius; k <= iradius; k++)
                for (int16_t j = -iradius; j <= iradius; j++)
                    for (int16_t i = -iradius; i <= iradius; i++)
                        if (histogramCondition(i, j, k, asqr, csqr))
                        {
                            if (tempArray(k + z, j + y, i + x) == 0)
                                numberOfDifs++;
                            tempArray(k + z, j + y, i + x) += origin + 1;
                        }
        }
        else
        {
            for (int16_t j = -iradius; j <= iradius; j++)
                for (int16_t i = -iradius; i <= iradius; i++)
                    if (i * i + j * j <= radius * radius)
                    {
                        if (tempArray(0, j + y, i + x) == 0)
                            numberOfDifs++;
                        tempArray(0, j + y, i + x) += origin + 1;
                    }
        }
    }

    Coords* DifferenceAdd = new Coords[numberOfDifs];   //indexes relative to new step pixel to add to histogram
    Coords* DifferenceRem = new Coords[numberOfDifs];   //indexes relative to new step pixel to remove from histogram
    *DifferenceAddPtr = DifferenceAdd;
    *DifferenceRemPtr = DifferenceRem;

    uint16_t addIndex = 0,
             remIndex = 0;

    if (threeDim)
        for (int16_t k = 0; k < margin; k++)
            for (int16_t j = 0; j < margin; j++)
                for (int16_t i = 0; i < margin; i++)
                {
                    if (tempArray(k, j, i) == 1)
                        DifferenceRem[remIndex++] = Coords(k - z, j - y, i - x);
                    if (tempArray(k, j, i) == 2)
                        DifferenceAdd[addIndex++] = Coords(k - z, j - y, i - x);
                }
    else
        for (int16_t j = 0; j < margin; j++)
            for (int16_t i = 0; i < margin; i++)
            {
                if (tempArray(0, j, i) == 1)
                    DifferenceRem[remIndex++] = Coords(0, j - y, i - x);
                if (tempArray(0, j, i) == 2)
                    DifferenceAdd[addIndex++] = Coords(0, j - y, i - x);
            }

    return numberOfDifs;
}


/// <summary>
/// Calculates domination of more intense pixels in histogram
/// </summary>
/// <param name="pixel">Currently considered input pixel</param>
/// <param name="histogram">Used histogram</param>
/// <param name="threshold"></param>
template<class InBitDepth, class OutBitDepth>
OutBitDepth CalculateDominanceOverMoreIntense(InBitDepth pixel, HistogramArray<OutBitDepth>& histogram, int threshold)
{
    if (pixel + threshold <= 0)
        return 0;

    OutBitDepth result = 0;

    for (uint32_t i = 0; i < pixel + threshold; i++) //add numbers of pixels that are >= pixel + threshold
        result += histogram[i];

    return result;
}

/// <summary>
/// Calculates number of less intense pixels in histogram
/// </summary>
/// <param name="pixel">Currently considered input pixel</param>
/// <param name="histogram">Used histogram</param>
/// <param name="threshold"></param>
template<class InBitDepth, class OutBitDepth>
OutBitDepth CalculateDominanceOverLessIntense(InBitDepth pixel, HistogramArray<OutBitDepth>& histogram, int threshold)
{
    OutBitDepth result = 0;
    uint32_t start = pixel + threshold > 0 ? pixel + threshold : 0;

    for (uint32_t i = start; i < histogram.Length(); i++) //add numbers of pixels that are >= pixel + threshold
        result += histogram[i];

    return result;
}

template<class InBitDepth, class OutBitDepth>
void FlyingHistogram(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, float radiusZ, int threshold, bool moreIntense, bool threeDim)
{
    uint32_t width  = image.Width(),
             height = image.Height(),
             frames = image.Frames();
    
    bool anisotropic = (radius != radiusZ) ? true : false;
    if (!threeDim)
        anisotropic = false;

    uint16_t iradius  = std::ceil(radius);
    uint16_t iradiusZ = std::ceil(radiusZ);

    auto CalculateDominance = moreIntense ? 
         CalculateDominanceOverLessIntense<InBitDepth, OutBitDepth>:
         CalculateDominanceOverMoreIntense<InBitDepth, OutBitDepth>;

    uint16_t diffLen = 0, diffLenZ = 0;
    Coords* DiffAddZ, * DiffRemZ, * DiffAddY, * DiffRemY, * DiffAddX, * DiffRemX;   //array of coords of delta pixels

    if (threeDim)
    {
        diffLenZ = SetUpRadiusDifference(radius, radiusZ, &DiffAddZ, &DiffRemZ, true, anisotropic, Direction::Z); //number of delta pixels
        diffLen  = SetUpRadiusDifference(radius, radiusZ, &DiffAddY, &DiffRemY, true, anisotropic, Direction::Y);
                   SetUpRadiusDifference(radius, radiusZ, &DiffAddX, &DiffRemX, true, anisotropic, Direction::X);
    }
    else
    {
        diffLen = SetUpRadiusDifference(radius, &DiffAddY, &DiffRemY, false, Direction::Y); //number of delta pixels
                  SetUpRadiusDifference(radius, &DiffAddX, &DiffRemX, false, Direction::X);
    }

    if (threeDim)
    {
        HistogramArray<OutBitDepth> histogramZ = HistogramArray<OutBitDepth>();

        float asqr = radius * radius;
        float csqr = radiusZ * radiusZ;

        auto histogramCondition = anisotropic ? anisotropicCondition : sphereCondition;

        for (int16_t k = -iradius; k <= iradius; k++)
        {
            for (int16_t j = -iradius; j <= iradius; j++)
                for (int16_t i = -iradius; i <= iradius; i++)
                    if (histogramCondition(i, j, k, asqr, csqr))
                        histogramZ[image(iradiusZ + k, iradius + j, iradius + i)]++; // compute first histogram
        }
        std::cout << "\n";
        for (uint32_t z = iradiusZ; z < frames - iradiusZ; z++)
        {
            std::cout << z << " ";
            if (z != iradiusZ)
                for (uint32_t i = 0; i < diffLenZ; i++)      // compute by removing and adding delta pixels to histogram
                {
                    histogramZ[image(z + DiffRemZ[i].z, iradius + DiffRemZ[i].y, iradius + DiffRemZ[i].x)]--;
                    histogramZ[image(z + DiffAddZ[i].z, iradius + DiffAddZ[i].y, iradius + DiffAddZ[i].x)]++;
                }

            HistogramArray<OutBitDepth> histogramY = HistogramArray<OutBitDepth>(histogramZ);

            for (uint32_t y = iradius; y < height - iradius; y++)
            {
                if (y != iradius)
                    for (uint32_t i = 0; i < diffLen; i++)
                    {
                        histogramY[image(z + DiffRemY[i].z, y + DiffRemY[i].y, iradius + DiffRemY[i].x)]--;
                        histogramY[image(z + DiffAddY[i].z, y + DiffAddY[i].y, iradius + DiffAddY[i].x)]++;
                    }

                HistogramArray<OutBitDepth> histogramX = HistogramArray<OutBitDepth>(histogramY);
                for (uint32_t x = iradius; x < width - iradius; x++)
                {
                    if (x != iradius)
                        for (uint32_t i = 0; i < diffLen; i++)
                        {
                            histogramX[image(z + DiffRemX[i].z, y + DiffRemX[i].y, x + DiffRemX[i].x)]--;
                            histogramX[image(z + DiffAddX[i].z, y + DiffAddX[i].y, x + DiffAddX[i].x)]++;
                        }
                    output(z, y, x) = CalculateDominance(image(z, y, x), histogramX, threshold);
                }
            }
        }
    }
    else
    {
        HistogramArray<OutBitDepth> histogramY = HistogramArray<OutBitDepth>();

        for (int16_t j = -iradius; j <= iradius; j++)
            for (int16_t i = -iradius; i <= iradius; i++)
                if (i * i + j * j <= radius * radius)
                    histogramY[image(0, iradius + j, iradius + i)]++; // compute first histogram

        for (uint32_t y = iradius; y < height - iradius; y++)
        {
            if (y != iradius)
                for (uint32_t i = 0; i < diffLen; i++)
                {
                    histogramY[image(0, y + DiffRemY[i].y, iradius + DiffRemY[i].x)]--;
                    histogramY[image(0, y + DiffAddY[i].y, iradius + DiffAddY[i].x)]++;
                }

            HistogramArray<OutBitDepth> histogramX = HistogramArray<OutBitDepth>(histogramY);
            for (uint32_t x = iradius; x < width - iradius; x++)
            {
                if (x != iradius)
                    for (uint32_t i = 0; i < diffLen; i++)
                    {
                        histogramX[image(0, y + DiffRemX[i].y, x + DiffRemX[i].x)]--;
                        histogramX[image(0, y + DiffAddX[i].y, x + DiffAddX[i].x)]++;
                    }
                output(0, y, x) = CalculateDominance(image(0, y, x), histogramX, threshold);
            }
        }
    }
    delete[] DiffAddZ, DiffRemZ, DiffAddY, DiffRemY, DiffAddX, DiffRemX;
}

template<class T1, class T2>
Image<T2> Compute(Image<T1>& input, float radius, int threshold, bool moreIntense, bool threeDim, bool onCpu, bool usingFH)
{
    Image<T2>output{ input.Width(), input.Height(), input.Frames() };

    auto start = std::chrono::high_resolution_clock::now();
    if (!onCpu)
    {
        if (!usingFH)
            GPU::SDAExt(input, output, radius, threshold, moreIntense);
        else
            GPU::FlyingHistogramExt(input, output, radius, threshold, moreIntense);
    }
    else
    {
        if (!usingFH)
            FlyingHistogram(input, output, radius, radius, threshold, moreIntense, threeDim);
        else
            if (threeDim)
                SDAborderless(input, output, radius, threshold);
            else
                SDAborderless2D(input, output, radius, threshold);
    }
    auto finish = std::chrono::high_resolution_clock::now();

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);

    std::cout << "\nTime Elapsed:" << milliseconds.count() << "ms\n";

    return output;
}

template <class T1, class T2>
bool SaveNormalized(Image<T1>& input, float radius, int threshold, bool moreIntense, bool threeDim, bool onCpu, bool usingFH, bool normalize, std::string outputFilename)
{
    if (radius < 24)
    {
        Image<uint16_t> output = Compute<T1, uint16_t>(input, radius, threshold, moreIntense, threeDim, onCpu, usingFH);
        
        if (!normalize)
            return SaveTiff(output, outputFilename.c_str());
        else
        {
            Image<T2> outputRescaled{ input.Width(), input.Height(), input.Frames() };
            if (sizeof(T2) >= sizeof(uint16_t))
            {
                outputRescaled.CopyValuesFrom(output);
                outputRescaled.Normalize();
            }
            else
            {
                output.Normalize(std::numeric_limits<T2>::max());
                outputRescaled.CopyValuesFrom(output);
            }
            return SaveTiff(outputRescaled, outputFilename.c_str());
        }
    }
    else
    {
        Image<uint32_t> output = Compute<T1, uint32_t>(input, radius, threshold, moreIntense, threeDim, onCpu, usingFH);

        if (!normalize)
            return SaveTiff(output, outputFilename.c_str());
        else
        {
            Image<T2> outputRescaled{ input.Width(), input.Height(), input.Frames() };
            if (sizeof(T2) >= sizeof(uint32_t))
            {
                outputRescaled.CopyValuesFrom(output);
                outputRescaled.Normalize();
            }
            else
            {
                output.Normalize(std::numeric_limits<T2>::max());
                outputRescaled.CopyValuesFrom(output);
            }
            return SaveTiff(outputRescaled, outputFilename.c_str());
        }
    }
}

int main(int argc, char** argv)
{
    std::string inputFilename;
    std::string outputFilename{"output.tiff"};
    float radius  = 0;
    float radiusZ = 0;
    int threshold = 0;
    bool help        = false;
    bool twoDim      = false;
    bool threeDim    = false;
    bool moreIntense = false;
    bool onCpu       = false;
    bool usingFH     = false;
    bool normalize   = false;
    bool out8        = false;
    bool out16       = false;
    bool out32       = false;

    sweet::Options opt(argc, const_cast<char**>(argv), "Description:");

    opt.get("-i",   "--input",        "Input filename",                                           inputFilename);
    opt.get("-o",   "--output",       "Output filename (optional)",                               outputFilename);

    opt.get("-r",   "--radius",       "Radius in pixels",                                         radius);
    opt.get("-z",   "--radius-z",     "Radius in pixels in anisotropic axis (optional)",          radiusZ);
    opt.get("-t",   "--threshold",    "Intensity threshold (optional)",                           threshold);
    opt.get("-m",   "--more-intense", "Calculate dominance over more intense pixels (optional)",  moreIntense);

    opt.get("-2",   "--2-dim",        "Two dimensional algorithm (CPU only)",                     twoDim);
    opt.get("-3",   "--3-dim",        "Three dimensional algorithm",                              threeDim);
    
    opt.get("-c",   "--cpu",          "Compute sequentially on CPU (optional)",                   onCpu);
    opt.get("-f",   "--fly-hist",     "Use flying histogram version of algorithm (optional)",     usingFH);

    opt.get("-n",   "--normalize",    "Normalize output to selected bit depth",                   normalize);
    opt.get("-o8",  "--output-8",     "Use 8-bit output",                                         out8);
    opt.get("-o16", "--output-16",    "Use 16-bit output",                                        out16);
    opt.get("-o32", "--output-32",    "Use 32-bit output",                                        out32);
   
    opt.get("-h",   "--help",         "Get help",                                                 help);

    opt.finalize();

    if (inputFilename.empty() || radius == 0.0)
    {
        if (!help)
            std::cerr << "Argument error! Run: \n" << argv[0] << " --help \nto see valid options.\n";
        return 1;
    }

    if (twoDim && threeDim)
    {
        std::cerr << "Argument error! Please select only one dimensionality (2D/3D)\n";
        return 1;
    }
    else if (!(twoDim || threeDim))
    {
        std::cerr << "Argument error! Please select one dimensionality (2D/3D)\n";
        return 1;
    }

    if (!moreIntense)
        threshold = -threshold;

    Image<uint8_t> input{};
    
    if (!ReadTiff(input, inputFilename.c_str()))
    {
        std::cerr << "\n Error while reading"s + inputFilename + "\Terminating.";
        return 1;
    }

    if (out8 || !normalize)
        SaveNormalized<uint8_t, uint8_t> (input, radius, threshold, moreIntense, threeDim, onCpu, usingFH, normalize, outputFilename);
    else if (out16)
        SaveNormalized<uint8_t, uint16_t>(input, radius, threshold, moreIntense, threeDim, onCpu, usingFH, normalize, outputFilename);
    else if (out32)
        SaveNormalized<uint8_t, uint32_t>(input, radius, threshold, moreIntense, threeDim, onCpu, usingFH, normalize, outputFilename);


    std::cout << "Finished" << std::endl;

    return 0;
}
