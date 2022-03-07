// SDAonCUDA.cpp : Defines the entry point for the application.
//

#include "SDAonCUDA.h"
#include "tinytiffreader.h"
#include "tinytiffwriter.h"

#include <cstring>
#include <chrono>

enum Direction
{
    Zdir = 0,
    Ydir = 1,
    Xdir = 2
};

template<class BitDepth>
class Image
{
public:
    uint32_t  width,
              height,
              frames;
    BitDepth* sample;

    Image()
    {
        width = 0;
        height = 0;
        frames = 0;
        sample = nullptr;
    }

    Image(Image& pattern)
    {
        width = pattern.width;
        height = pattern.height;
        frames = pattern.frames;
        sample = (BitDepth*)calloc(frames * (uint64_t)height * width, sizeof(BitDepth));
        //sample = new BitDepth[frames * (uint64_t)height * width];
    }

    Image(int _width, int _height, int _frames)
    {
        width = _width;
        height = _height;
        frames = _frames;
        sample = (BitDepth*)calloc(frames * (uint64_t)height * width, sizeof(BitDepth));
        //sample = new BitDepth[frames * (uint64_t)height * width];
    }

    ~Image()
    {
        free(sample);
    }

    BitDepth& Image::operator()(uint32_t z, uint32_t y, uint32_t x)
    {
        return sample[(z * height + y) * width + x];
    }

    uint64_t Index(uint32_t z, uint32_t y, uint32_t x) const
    {
        return (z * (uint64_t)height + y) * width + x;
    }


    void SetSize(uint32_t _width, uint32_t _height, uint32_t _frames)
    {
        if (sample != nullptr)
            free(sample);
        width = _width;
        height = _height;
        frames = _frames;
        sample = (BitDepth*)calloc(frames * (uint64_t)height * width, sizeof(BitDepth));
    }

    /// <summary>
    /// Deep copies single slide into image object.
    /// </summary>
    /// <param name="frame">index of frame being saved</param>
    /// <param name="newslide">data to save</param>
    /// <returns></returns>
    bool SetSlide(uint32_t frame, BitDepth* newslide)
    {
        if (frame < frames)
        {
            BitDepth* slidestart = (sample + frame * (uint64_t)width * height);
            for (uint32_t i = 0; i < width * height; i++)
            {
                slidestart[i] = newslide[i];
            }
        }
        else
            return false;
        return true;
    }


    /*void SetTo(BitDepth value)
    {
        for (uint32_t i = 0; i < width * (uint64_t)height * frames; i++)
            sample[i] = value;
    }*/


    BitDepth MaxValue()
    {
        BitDepth max = sample[0];
        for (uint32_t z = 0; z < frames; z++)
            for (uint32_t y = 0; y < height; y++)
                for (uint32_t x = 0; x < width; x++)
                    if (max < sample[Index(z, y, x)])
                        max = sample[Index(z, y, x)];
        return max;
    }

    void Normalize()
    {
        BitDepth max = MaxValue();
        max = 1;
        uint32_t newMax = (1 << sizeof(BitDepth) * 8) - 1;    // 2^size -1
        for (uint32_t z = 0; z < frames; z++)
            for (uint32_t y = 0; y < height; y++)
                for (uint32_t x = 0; x < width; x++)
                    sample[Index(z, y, x)] = (sample[Index(z, y, x)] * newMax) / max;
    }

    void Clear()
    {
        for (uint32_t z = 0; z < frames; z++)
            for (uint32_t y = 0; y < height; y++)
                for (uint32_t x = 0; x < width; x++)
                    sample[Index(z, y, x)] = 0;
    }

    bool Image::operator==(Image& rhs) const
    {
        if (frames == rhs.frames && height == rhs.height && width == rhs.width)
        {
            for (uint32_t z = 0; z < frames; z++)
                for (uint32_t y = 0; y < height; y++)
                    for (uint32_t x = 0; x < width; x++)
                        if (sample[Index(z, y, x)] != rhs(z, y, x))
                            return false;
            return true;
        }
        else
            return false;
    }
};


class Coords
{
public:
    int z, y, x;

    Coords()
    {
        z = 0;
        y = 0;
        x = 0;
    }

    Coords(int _z, int _y, int _x)
    {
        z = _z;
        y = _y;
        x = _x;
    }

    Coords(Coords& pattern)
    {
        z = pattern.z;
        y = pattern.y;
        x = pattern.x;
    }

    bool Coords::operator==(const Coords& rhs)
    {
        return (z == rhs.z && y == rhs.y && x == rhs.x);
    }
    bool Coords::operator!=(Coords& other)
    {
        return !(*this == other);
    }

    void Rotate90Z()
    {
        int temp = x;
        x = y;
        y = -temp;
    }
    void Rotate90y()
    {
        int temp = x;
        x = z;
        z = -temp;
    }
    void Rotate90x()
    {
        int temp = y;
        y = z;
        z = -temp;
    }

    void dPrint()
    {
        std::cout << "z:" << z << "\ty:" << y << "\tx:" << x << std::endl;
    }
};


class HistogramArray
{
    uint16_t* histogram;
    uint32_t length;
public:
    HistogramArray(uint16_t byteDepth)
    {
        length = 1 << 8 * byteDepth; // 2 ^ (bitDepth), so cell for every possible value
        histogram = new uint16_t[length];
        for (uint16_t i = 0; i < length; i++)
            histogram[i] = 0;
    }

    HistogramArray(const HistogramArray &pattern)
    {
        length = pattern.length;
        histogram = new uint16_t[length];
        for (uint16_t i = 0; i < length; i++)
            histogram[i] = pattern.histogram[i];
    }

    uint16_t& HistogramArray::operator()(uint16_t i)
    {
        return histogram[i];
    }
    
    uint32_t Length()
    {
        return length;
    }

    void Clear()
    {
        for (uint16_t i = 0; i < length; i++)
            histogram[i] = 0;
    }

    void dPrint()
    {
        std::cout << '\n';
        for (uint16_t i = 0; i < length; i++)
            std::cout << i << ": " << histogram[i] << '\t';
    }

    void dPrintSum()
    {
        uint32_t sum = 0;
        for (uint16_t i = 0; i < length; i++)
            sum += histogram[i];
        std::cout << "\nSum:" << sum << ' ';
    }
};

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
        BitDepth* slide = (BitDepth*)calloc(width * (uint64_t)height, sizeof(BitDepth));
        image.SetSize(width, height, frames);

        if (TinyTIFFReader_wasError(tiffr)) std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";
        else ok = true;

        for (uint32_t frame = 0; ok; frame++)
        {
            //std::cout << "frame: " << frame << std::endl;
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

        std::cout << "    read and checked all frames: " << ((ok) ? std::string("SUCCESS") : std::string("ERROR")) << " \n";
    }
    TinyTIFFReader_close(tiffr);

    return true;
}

template<class BitDepth>
bool SaveTiff(Image<BitDepth>& image, const char* filename)
{
    TinyTIFFWriterFile* tiff = TinyTIFFWriter_open(filename, 8, TinyTIFFWriter_UInt, 1, image.width, image.height, TinyTIFFWriter_Greyscale);
    //bits per sample constant cause of some errors

    std::cout << "\nSaving as '" << filename << "'\n";
    if (tiff)
    {
        for (size_t f = 0; f < image.frames; f++)
        {
            int res = TinyTIFFWriter_writeImage(tiff, image.sample + (f * image.width * image.height)); //TinyTIFF_Planar   TinyTIFF_Chunky
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

template<class InBitDepth, class OutBitDepth>
void SDA(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold)
{
    uint32_t width = image.width,
             height = image.height,
             frames = image.frames;
    uint16_t iradius = (uint16_t)(radius + 0.999);  //cheaply ceiled radius

    for (uint32_t z = iradius; z < frames - iradius; z++)
    {
        std::cout << z << " ";
        for (uint32_t y = iradius; y < height - iradius; y++)
            for (uint32_t x = iradius; x < width - iradius; x++)
                for (int16_t k = -iradius; k <= iradius; k++)
                    for (int16_t j = -iradius; j <= iradius; j++)
                        for (int16_t i = -iradius; i <= iradius; i++)
                            if (i * i + j * j + k * k <= radius * radius)
                                if (image(z + k, y + j, x + i) >= image(z, y, x) + threshold)
                                    output(z, y, x)++;
    }
}

template<class InBitDepth, class OutBitDepth>
void SDAborderless(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold)
{
    uint32_t width = image.width,
             height = image.height,
             frames = image.frames;
    uint16_t iradius = (uint16_t)(radius + 0.999);  //cheaply ceiled radius

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

//template<class InBitDepth, class OutBitDepth>
//void SinglePixelDominance(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold, uint32_t z, uint32_t y, uint32_t x)
//{
//    for (int16_t k = -iradius; k <= iradius; k++)
//        if (0 <= z + k && z + k < frames)
//            for (int16_t j = -iradius; j <= iradius; j++)
//                if (0 <= y + j && y + j < height)
//                    for (int16_t i = -iradius; i <= iradius; i++)
//                        if (i * i + j * j + k * k <= radius * radius && 0 <= x + i && x + i < width)
//                            if (image(z + k, y + j, x + i) >= image(z, y, x) + threshold)
//                                output(z, y, x)++;
//}

/// <summary>
/// 
/// </summary>
/// <param name="radius"></param>
/// <param name="direction"> 0 - z, 1 - y, 2 - x</param>
/// <param name="DifferenceAddPtr"></param>
/// <param name="DifferenceRemPtr"></param>
/// <returns></returns>
uint16_t SetUpRadiusDifference(float radius, Coords** DifferenceAddPtr, Coords** DifferenceRemPtr)
{
    uint16_t iradius = (uint16_t)(radius + 0.999);  //cheaply ceiled radius
    uint16_t margin = iradius * 2 + 2;              //to fit 2 offset spheres
    uint16_t numberOfDifs = 0;                      //number of delta pixels

    Image<uint8_t> tempArray = Image<uint8_t>(margin, margin, margin);  //sandbox
    uint16_t z = iradius, 
             y = iradius, 
             x = iradius;

    for (uint8_t origin = 0; origin < 2; origin++)  //origin offset + id to mark 2 offset spheres
    {
        z = iradius + origin;
        numberOfDifs = 0;
        for (int16_t k = -iradius; k <= iradius; k++)
            for (int16_t j = -iradius; j <= iradius; j++)
                for (int16_t i = -iradius; i <= iradius; i++)
                    if (i * i + j * j + k * k <= radius * radius)
                    {
                        if (tempArray(k + z, j + y, i + x) == 0)
                            numberOfDifs++;
                        tempArray(k + z, j + y, i + x) += origin + 1;
                    }
    }

    //std::cout << numberOfDifs << " ";

    Coords* DifferenceAdd = new Coords[numberOfDifs];   //indexes relative to new step pixel to add to histogram
    Coords* DifferenceRem = new Coords[numberOfDifs];   //indexes relative to new step pixel to remove from histogram
    *DifferenceAddPtr = DifferenceAdd;
    *DifferenceRemPtr = DifferenceRem;

    uint16_t addIndex = 0,
             remIndex = 0;
    //uint16_t debugHistVol = 0;

    for (int16_t k = 0; k < margin; k++)
        for (int16_t j = 0; j < margin; j++)
            for (int16_t i = 0; i < margin; i++)
            {
                if (tempArray(k, j, i) == 1)
                    DifferenceRem[remIndex++] = Coords(k - z, j - y, i - x);
                if (tempArray(k, j, i) == 2)
                    DifferenceAdd[addIndex++] = Coords(k - z, j - y, i - x);
                //if (tempArray(k, j, i) == 3)
                //    debugHistVol++;
            }
    //debugHistVol += numberOfDifs;

    return numberOfDifs;
}

/// <summary>
/// 
/// </summary>
/// <param name="pixel">Currently considered input pixel</param>
/// <param name="histogram">Used histogram</param>
/// <param name="threshold"></param>
template<class InBitDepth, class OutBitDepth>
OutBitDepth CalculateDominance(InBitDepth pixel, HistogramArray& histogram, int threshold)
{
    OutBitDepth result = 0;

    for (uint32_t i = pixel + threshold; i < histogram.Length(); i++) //add numbers of pixels that are >= pixel + threshold
    {
        uint16_t volume = histogram(i);
        result += volume;
    }
    return result;
}

template<class InBitDepth, class OutBitDepth>
void FlyingHistogram(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold)
{
    uint32_t width = image.width,
             height = image.height,
             frames = image.frames;
    uint16_t iradius = (uint16_t)(radius + 0.999);  //cheaply ceiled radius

    Coords* DiffAddZ, * DiffRemZ;  //array of coords of delta pixels
    uint16_t DiffLen = SetUpRadiusDifference(radius, &DiffAddZ, &DiffRemZ); //number of delta pixels
    Coords* DiffAddY = new Coords[DiffLen], * DiffRemY = new Coords[DiffLen],
          * DiffAddX = new Coords[DiffLen], * DiffRemX = new Coords[DiffLen];

    for (uint16_t i = 0; i < DiffLen; i++)  // just rotate vectors instead of generating new
    {
        DiffAddX[i] = DiffAddY[i] = DiffAddZ[i]; 
        DiffRemX[i] = DiffRemY[i] = DiffRemZ[i];
        DiffAddY[i].Rotate90x();
        DiffRemY[i].Rotate90x();
        DiffAddX[i].Rotate90y();
        DiffRemX[i].Rotate90y();
    }

    HistogramArray histogramZ = HistogramArray(sizeof(OutBitDepth));

    //debug logic check
    //for (uint32_t z = iradius; z < iradius + 3; z++)
    //    for (uint32_t y = iradius; y < height - iradius; y++)
    //        for (uint32_t x = iradius; x < width - iradius; x++)
    //        {
    //            histogramZ.Clear();
    //            for (int16_t k = -iradius; k <= iradius; k++)
    //                for (int16_t j = -iradius; j <= iradius; j++)
    //                    for (int16_t i = -iradius; i <= iradius; i++)
    //                        if (i * i + j * j + k * k <= radius * radius)
    //                            histogramZ(image(z + k, y + j, x + i))++;
    //            histogramZ.dPrintSum();
    //            output(z, y, x) = CalculateDominance<InBitDepth, OutBitDepth>(image(z, y, x), histogramZ, threshold);
    //        }

    for (int16_t k = -iradius; k <= iradius; k++)
        for (int16_t j = -iradius; j <= iradius; j++)
            for (int16_t i = -iradius; i <= iradius; i++)
                if (i * i + j * j + k * k <= radius * radius)
                    histogramZ(image(iradius + k, iradius + j, iradius + i))++; // compute first histogram
    //histogramZ.dPrintSum();


    for (uint32_t z = iradius; z < frames - iradius; z++)
    {
        if(z != iradius)
            for (uint32_t i = 0; i < DiffLen; i++)      // compute by removing and adding delta pixels to histogram
            {
                histogramZ(image(z + DiffRemZ[i].z, iradius + DiffRemZ[i].y, iradius + DiffRemZ[i].x))--;
                histogramZ(image(z + DiffAddZ[i].z, iradius + DiffAddZ[i].y, iradius + DiffAddZ[i].x))++;
            }

        HistogramArray histogramY = HistogramArray(histogramZ);
        for (uint32_t y = iradius; y < height - iradius; y++)
        {
            if (y != iradius)
                for (uint32_t i = 0; i < DiffLen; i++)
                {
                    histogramY(image(z + DiffRemY[i].z, y + DiffRemY[i].y, iradius + DiffRemY[i].x))--;
                    histogramY(image(z + DiffAddY[i].z, y + DiffAddY[i].y, iradius + DiffAddY[i].x))++;
                }

            HistogramArray histogramX = HistogramArray(histogramY);
            for (uint32_t x = iradius; x < width - iradius; x++)
            {
                if (x != iradius)
                    for (uint32_t i = 0; i < DiffLen; i++)
                    {
                        histogramX(image(z + DiffRemX[i].z, y + DiffRemX[i].y, x + DiffRemX[i].x))--;
                        histogramX(image(z + DiffAddX[i].z, y + DiffAddX[i].y, x + DiffAddX[i].x))++;
                    }
                output(z, y, x) = CalculateDominance<InBitDepth, OutBitDepth>(image(z, y, x), histogramX, threshold);
            }
        }
    }
}



int main()
{
    std::string file = "C:/Users/Miko/Desktop/MgrTif/";
    std::string input = "zebraCropped 30x30x5";
    std::string output = "zc 30x30x5 test d";
    std::string type = ".tif";

    std::cout << "Started\n";

    ///////// ensure correctness of rotation
    //float radius = 3.5;
    //
    //Coords* DiffAddZ, * DiffRemZ, * DiffAddY, * DiffRemY, * DiffAddX, * DiffRemX;  //array of coords of delta pixels
    //uint16_t DiffLen = SetUpRadiusDifference(radius, Zdir, &DiffAddZ, &DiffRemZ); //number of delta pixels
    //SetUpRadiusDifference(radius, Ydir, &DiffAddY, &DiffRemY);
    //SetUpRadiusDifference(radius, Xdir, &DiffAddX, &DiffRemX);
    //
    //bool same;
    //for (size_t i = 0; i < DiffLen; i++)
    //{
    //    DiffAddZ[i].Rotate90x();
    //    same = false;
    //    for (size_t j = 0; j < DiffLen; j++)
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

    ReadTiff(croppedImage, (file + input + type).c_str());

    for (int i = 0; i < 1; i++)
    {
        float radius = 2.5;
        int thresh = 40 + 5 * i;
        Image<uint8_t> out = Image<uint8_t>(croppedImage);
        Image<uint8_t> out2 = Image<uint8_t>(croppedImage);

        auto start = std::chrono::high_resolution_clock::now();
        SDA(croppedImage, out, radius, thresh);
        auto finish = std::chrono::high_resolution_clock::now();

        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
        std::cout << "\nTime Elapsed:" << microseconds.count() << "us\n";


        auto startFH = std::chrono::high_resolution_clock::now();
        FlyingHistogram(croppedImage, out2, radius, thresh);
        auto finishFH = std::chrono::high_resolution_clock::now();

        auto microsecondsFH = std::chrono::duration_cast<std::chrono::microseconds>(finishFH - startFH);
        std::cout << "\nTime Elapsed:" << microsecondsFH.count() << "us\n";

        if (out == out2)
            std::cout << "\nSame outputs\n";
        else
            std::cout << "\nDifferent outputs\n";


        out.Normalize();
        char buffer[128] = { 0 };
        sprintf_s(buffer, "%s %i - %i %s", (file + output).c_str(), (int)(radius * 100), thresh, type.c_str());
        std::cout << '\n' << buffer << '\n';
        SaveTiff(out, buffer);
    }

    std::cout << "Finished\n";
    return 0;
}
