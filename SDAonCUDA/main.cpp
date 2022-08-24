// SDAonCUDA.cpp : Defines the entry point for the application.
//

#include <cstring>
#include <chrono>
#include <cmath>

#include "main.h"
#include "tinytiffreader.h"
#include "tinytiffwriter.h"
#include "cuda.h"


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
    memset(data, 0, GetSize());
}

template<class BitDepth>
Image<BitDepth>::Image(int _width, int _height, int _frames)
{
    width = _width;
    height = _height;
    frames = _frames;
    data = new BitDepth[GetSize()];
    memset(data, 0, GetSize());
}


template<class BitDepth>
Image<BitDepth>::~Image()
{
    delete[] data;
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
        memcpy(slidestart, newslide, width * static_cast<uint64_t>(height));
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
        max = 1;
    uint32_t newMax = (uint32_t)(std::numeric_limits<BitDepth>::max);
    for (BitDepth* p = data; p < data + GetSize(); ++p)
        *p = (*p * newMax) / max;
}


template<class BitDepth>
void Image<BitDepth>::Clear()
{
    memset(data, 0, GetSize());
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
    std::cout << "\nReading '" << std::string(filename) << "'\n";

    TinyTIFFReaderFile* tiffr = TinyTIFFReader_open(filename);
    if (!tiffr) {
        std::cout << "    ERROR reading (not existent, not accessible or no TIFF file)\n";
        return false;
    }
    else {
        if (TinyTIFFReader_wasError(tiffr)) std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";

        uint32_t width = TinyTIFFReader_getWidth(tiffr);
        uint32_t height = TinyTIFFReader_getHeight(tiffr);
        uint32_t frames = TinyTIFFReader_countFrames(tiffr);
        BitDepth* slide = new BitDepth[width * static_cast<uint64_t>(height)];
        image.SetSize(width, height, frames);

        if (TinyTIFFReader_wasError(tiffr)) std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";
        else ok = true;

        for (uint32_t frame = 0; ok; frame++)
        {
            TinyTIFFReader_getSampleData(tiffr, slide, 0);
            image.SetSlide(frame, slide);

            if (TinyTIFFReader_wasError(tiffr))
            {
                ok = false;
                std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";
            }
            if (!TinyTIFFReader_readNext(tiffr))
                break;
        }
        delete[] slide;
        std::cout << "    read and checked all frames: " << ((ok) ? std::string("SUCCESS") : std::string("ERROR")) << " \n";
    }
    TinyTIFFReader_close(tiffr);

    return true;
}

template<class BitDepth>
bool SaveTiff(Image<BitDepth>& image, const char* filename)
{
    TinyTIFFWriterFile* tiff = TinyTIFFWriter_open(filename, 8, TinyTIFFWriter_UInt, 1, image.Width(), image.Height(), TinyTIFFWriter_Greyscale);
    //bits per sample constant cause of some errors

    std::cout << "\nSaving as '" << filename << "'\n";
    if (tiff)
    {
        for (size_t f = 0; f < image.Frames(); f++)
        {
            int res = TinyTIFFWriter_writeImage(tiff, image.GetDataPtr() + (f * image.Width() * image.Height())); //TinyTIFF_Planar   TinyTIFF_Chunky
            if (res != TINYTIFF_TRUE)
            {
                std::cout << "ERROR: error writing image data into '" << filename << "'! MESSAGE: " << TinyTIFFWriter_getLastError(tiff) << "\n";
                TinyTIFFWriter_close(tiff);
                return false;
            }
        }
        TinyTIFFWriter_close(tiff);
        std::cout << "File saved as '" << filename << "'\n";
        return true;
    }
    std::cout << "ERROR: could not open '" << filename << "' for writing!\n";
    return false;
}

template<class BitDepth>
bool CropTiff(Image<BitDepth>& image, Image<BitDepth>& croppedImage,
    uint32_t width0, uint32_t height0, uint32_t frames0,
    uint32_t width1, uint32_t height1, uint32_t frames1)
{
    if (width1 < width0 || height1 < height0 || frames1 < frames0)
        return false;
    if (image.width < width1 || image.height < height1 || image.frames < frames1)
        return false;
    croppedImage.SetSize(width1 - width0, height1 - height0, frames1 - frames0);

    uint32_t width = image.width,
        height = image.height,
        frames = image.frames,
        newWidth = croppedImage.width,
        newHeight = croppedImage.height,
        newFrames = croppedImage.frames;

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
    uint32_t width = image.width,
             height = image.height,
             frames = image.frames;
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
    uint32_t width = image.width,
             height = image.height,
             frames = image.frames;
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
    uint32_t width = image.width,
        height = image.height;
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
            std::cout << k << " ";
            for (int16_t j = -iradius; j <= iradius; j++)
                for (int16_t i = -iradius; i <= iradius; i++)
                    if (histogramCondition(i, j, k, asqr, csqr))
                        histogramZ[image(iradiusZ + k, iradius + j, iradius + i)]++; // compute first histogram
        }
        std::cout << "\n";
        for (uint32_t z = iradiusZ; z < frames - iradius; z++)
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

int main()
{
    std::string file = "C:/Users/Miko/Desktop/MgrTif/";
    std::string input = "zebraCropped 50x50x5";
    std::string output = "zc 50x50x5 test";
    std::string type = ".tif";

    std::cout << "Started\n";

    ///////// ensure correctness of rotation
    //float radius = 3.5;
    //
    //Coords* DiffAddZ, * DiffRemZ, * DiffAddY, * DiffRemY, * DiffAddX, * DiffRemX;  //array of coords of delta pixels
    //uint16_t diffLen = SetUpRadiusDifference(radius, Z, &DiffAddZ, &DiffRemZ); //number of delta pixels
    //SetUpRadiusDifference(radius, Y, &DiffAddY, &DiffRemY);
    //SetUpRadiusDifference(radius, X, &DiffAddX, &DiffRemX);
    //
    //bool same;
    //for (size_t i = 0; i < diffLen; i++)
    //{
    //    DiffAddZ[i].Rotate90x();
    //    same = false;
    //    for (size_t j = 0; j < diffLen; j++)
    //    {
    //        if (DiffAddZ[i] == DiffAddY[j])
    //        {
    //            same = true;
    //            break;
    //        }
    //    }
    //    if (same)
    //        std::cout << "\nsame";
    //    else
    //        std::cout << "\ndifferent";
    //}

    Image<uint8_t> croppedImage = Image<uint8_t>();

    //Image<uint8_t> image = Image<uint8_t>();
    //ReadTiff(image, (file + input + type).c_str());
    //
    //if(!CropTiff(image, croppedImage, 600, 600, 0, 900, 900, 50))
    //{
    //    std::cout << "Failed Cropping image.\n";
    //    return 1;
    //}
    //SaveTiff(croppedImage, (file + output + type).c_str());

    ////gpu check 
    //int* a,* b,* c, len = 100;
    //a = new int[len];
    //b = new int[len];
    //c = new int[len];
    //for (int i = 0; i < len; i++)
    //{
    //    a[i] = 5 + i;
    //    b[i] = 3 - i;
    //    c[i] = 0;
    //}
    //Test::addWithCuda(c, a, b, len);
    //for (int i = 0; i < len; i++)
    //    std::cout << " " << c[i];
    //return 0;

    ReadTiff(croppedImage, (file + input + type).c_str());

    for (int i = 0; i < 1; i++)
    {
        float radius = 5;
        float radiusZ = 5;
        int thresh = 40;
        Image<uint8_t> out = Image<uint8_t>(croppedImage);
        Image<uint8_t> out2 = Image<uint8_t>(croppedImage);
        Image<uint8_t> out3 = Image<uint8_t>(croppedImage);

        auto start = std::chrono::high_resolution_clock::now();
        GPU::SDA(croppedImage, out, radius, thresh);
        //GPU::FlyingHistogram(croppedImage, out, radius, thresh, true);
        auto finish = std::chrono::high_resolution_clock::now();

        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << "\nTime Elapsed:" << milliseconds.count() << "ms\n";

        //auto startSDA = std::chrono::high_resolution_clock::now();
        //SDAborderless(croppedImage, out2, radius, thresh);
        //auto finishSDA = std::chrono::high_resolution_clock::now();

        //auto millisecondsSDA = std::chrono::duration_cast<std::chrono::milliseconds>(finishSDA - startSDA);
        //std::cout << "\nTime Elapsed:" << millisecondsSDA.count() << "ms\n";

        auto startFH = std::chrono::high_resolution_clock::now();
        //FlyingHistogram(croppedImage, out3, radius, thresh);
        FlyingHistogram(croppedImage, out3, radius, radiusZ, thresh, true, true);
        auto finishFH = std::chrono::high_resolution_clock::now();

        auto millisecondsFH = std::chrono::duration_cast<std::chrono::milliseconds>(finishFH - startFH);
        std::cout << "\nTime Elapsed:" << millisecondsFH.count() << "ms\n";

        //if (out == out2)\
            std::cout << "\nSame outputs\n";\
        else\
            std::cout << "\nDifferent outputs\n";
        if (out == out3)
            std::cout << "\nSame outputs\n";
        else
            std::cout << "\nDifferent outputs\n";

        //out.Normalize();
        std::cout << "\ndebug sum: " << out.dGetSum();
        char buffer[128] = { 0 };
        sprintf_s(buffer, "%s qqq%i - %i %s", (file + output).c_str(), (int)(radius * 100), thresh, type.c_str());
        std::cout << '\n' << buffer << '\n';
        SaveTiff(out, buffer);
    }

    std::cout << "Finished\n";
    return 0;
}
