// SDAonCUDA.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>

enum class Direction
{
    Z = 0,
    Y = 1,
    X = 2
};

struct Coords
{
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

    Coords(const Coords& pattern)
    {
        z = pattern.z;
        y = pattern.y;
        x = pattern.x;
    }

    bool Coords::operator==(const Coords& rhs)
    {
        return (z == rhs.z && y == rhs.y && x == rhs.x);
    }

    bool Coords::operator!=(const Coords& rhs)
    {
        return !(*this == rhs);
    }

    void Rotate90z()
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
};

template<class BitDepth>
class Image
{
private:
    uint32_t  width,
              height,
              frames;
    BitDepth* data;

public:
    Image();
    Image(const Image& pattern);
    Image(const Image& pattern, BitDepth val);
    Image(uint32_t _width, uint32_t _height, uint32_t _frames);
    ~Image();

    Image& operator=(const Image& pattern);

    uint32_t Width()  { return width;  };
    uint32_t Height() { return height; };
    uint32_t Frames() { return frames; };

    BitDepth* GetDataPtr() { return data; };
    BitDepth& Image::operator()(uint32_t z, uint32_t y, uint32_t x);
    bool operator==(Image& rhs) const;

    void Normalize();
    void Normalize(size_t newMax);
    uint64_t GetSize() const;
    void SetSize(uint32_t _width, uint32_t _height, uint32_t _frames);
    bool SetSlide(uint32_t frame, BitDepth* newslide);

    uint64_t dGetSum() const
    {
        uint64_t sum = 0;
        for (BitDepth* p = data; p < data + GetSize(); ++p)
        {
            sum += *p;
        }
        return sum;
    }

    template<class T2>
    void CopyValuesFrom(Image<T2>& pattern)
    {
        if (GetSize() != pattern.GetSize())
            return;
        //T2* q = pattern.GetDataPtr();\
        for (BitDepth* p = data; p < data + GetSize(); ++p)\
        {\
            *p = static_cast<BitDepth>(*q);\
            ++q;\
        }
        for (size_t k = 0; k < frames; k++)
            for (size_t j = 0; j < height; j++)
                for (size_t i = 0; i < width; i++)
                    data[Index(k, j, i)] = pattern(k, j, i);
    }

private:
    uint64_t Index(uint32_t z, uint32_t y, uint32_t x) const;
    BitDepth MaxValue() const;
    void Clear();
};

template<class BitDepth>
class HistogramArray
{
    uint16_t* histogram;
    uint32_t  length;
public:
    HistogramArray()
    {
        length = 1 + (uint32_t)std::numeric_limits<BitDepth>::max();
        histogram = new uint16_t[length];
        memset(histogram, 0, length * sizeof(*histogram));
    }

    HistogramArray(const HistogramArray& pattern)
    {
        length = pattern.length;
        histogram = new uint16_t[length];
        memcpy(histogram, pattern.histogram, length * sizeof(*histogram));
    }

    ~HistogramArray()
    {
        delete[] histogram;
    }

    uint16_t& HistogramArray::operator[](uint16_t i) const
    {
        return histogram[i];
    }

    uint32_t Length() const
    {
        return length;
    }

    void Clear()
    {
        memset(histogram, 0, length * sizeof(length));
    }

    uint64_t Sum()
    {
        uint64_t sum = 0;
        for (size_t i = 0; i < length; i++)
            sum += histogram[i];
        return sum;
    }
};

/// <summary>
/// Generates Coords, and counts number of deltapixels, by selected type
/// </summary>
/// <param name="radius"></param>
/// <param name="radiusZ"></param>
/// <param name="DifferenceAddPtr">array of delta pixels to add</param>
/// <param name="DifferenceRemPtr">array of delta pixels to remove</param>
/// <param name="threeDim">is three dimensional</param>
/// <param name="anisotropic">is anisotropic</param>
/// <param name="dir">direction of delta</param>
/// <returns>Number of delta pixels</returns>
uint16_t SetUpRadiusDifference(float radius, float radiusZ, Coords** DifferenceAddPtr, Coords** DifferenceRemPtr, bool threeDim, bool anisotropic, Direction dir);

/// <summary>
/// Generates Coords, and counts number of deltapixels, by selected type (non-anisotropic)
/// </summary>
/// <param name="radius"></param>
/// <param name="DifferenceAddPtr">array of delta pixels to add</param>
/// <param name="DifferenceRemPtr">array of delta pixels to remove</param>
/// <param name="threeDim">is three dimensional</param>
/// <param name="dir">direction of delta</param>
/// <returns>Number of delta pixels</returns>
uint16_t SetUpRadiusDifference(float radius, Coords** DifferenceAddPtr, Coords** DifferenceRemPtr, bool threeDim, Direction dir);

