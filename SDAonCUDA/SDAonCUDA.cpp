﻿// SDAonCUDA.cpp : Defines the entry point for the application.
//

#include "SDAonCUDA.h"
#include "tinytiffreader.h"
#include "tinytiffwriter.h"

#include <cstring>
#include <chrono>


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

};

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

template<class InBitDepth, class OutBitDepth>
void SDA(Image<InBitDepth>& image, Image<OutBitDepth>& output, float radius, int threshold)
{
    uint32_t width = image.width,
             height = image.height,
             frames = image.frames;
    uint16_t iradius = (uint16_t)(radius + 0.999);  //cheap ceil

    for (uint32_t z = iradius; z < frames - iradius; z++)
    {
        std::cout << z << " ";
        for (uint32_t y = iradius; y < height - iradius; y++)
            for (uint32_t x = iradius; x < width - iradius; x++)
                for (int16_t k = -iradius; k <= iradius; k++)
                    for (int16_t i = -iradius; i <= iradius; i++)
                        for (int16_t j = -iradius; j <= iradius; j++)
                            if (i * i + j * j + k * k <= radius * radius)
                                if (image(z + k, y + i, x + j) >= image(z, y, x) + threshold)
                                    output(z, y, x)++;
    }
}



int main()
{
    std::string file = "C:/Users/Miko/Desktop/MgrTif/";
    std::string input = "zebrafish";
    std::string output = "zebraCropped 10x10x1";
    std::string type = ".tif";

    std::cout << "Started\n";

    Image<uint8_t> image = Image<uint8_t>();
    ReadTiff(image, (file + input + type).c_str());

    Image<uint8_t> croppedImage = Image<uint8_t>();
    if(!CropTiff(image, croppedImage, 600, 600, 0, 900, 900, 50))
    {
        std::cout << "Failed Cropping image.\n";
        return 1;
    }
    SaveTiff(croppedImage, (file + output + type).c_str());
    //ReadTiff(croppedImage, "C:/Users/Miko/Desktop/MgrTif/zebraCropped 5x5x1.tif");

    for (int i = 0; i < 1; i++)
    {
        float radius = 3.5;
        int thresh = 40 + 5 * i;
        Image<uint8_t> out = Image<uint8_t>(croppedImage.width, croppedImage.height, croppedImage.frames);

        auto start = std::chrono::high_resolution_clock::now();
        SDA(croppedImage, out, radius, thresh);
        auto finish = std::chrono::high_resolution_clock::now();

        std::cout << "\nTime Elapsed:" << (finish - start).count() << std::endl;

        out.Normalize();
        char buffer[128] = { 0 };
        sprintf_s(buffer, "%s%s %i - %i %s", file.c_str(), output.c_str(), (int)(radius * 100), thresh, type.c_str());
        std::cout << '\n' << buffer << '\n';
        SaveTiff(out, buffer);
    }

    std::cout << "Finished\n";
    return 0;
}
