#pragma once

#include <memory>
#include <stdlib.h>
#include <stdio.h>
#include <png.h>
#include "io.h"
#include "platform.h"

class StaticBuffer
{
public:
	StaticBuffer() : m_size(0), data(nullptr) {};
	StaticBuffer(size_t size) : m_size(0), data(nullptr)
	{
		data = std::unique_ptr<unsigned char[]>(new unsigned char[size]);
		m_size = size;
	};
	StaticBuffer& operator=(StaticBuffer&& other)
	{
		data = std::move(other.data);
		m_size = other.m_size;
		return *this;
	}

	size_t size() const { return m_size; };

public:
	std::unique_ptr<unsigned char[]> data;

private:
	size_t m_size;
};

class Image
{
public:
	Image() {};
	Image(size_t width, size_t height, int channeldepth, int channels)
		: m_width(width),
		m_height(height),
		m_channeldepth(channeldepth),
		m_channels(channels)
	{
		size_t size = m_width * m_height * pixelsize();
		m_data = std::shared_ptr<StaticBuffer>(new StaticBuffer(size));
	}

	unsigned char* data() { return m_data->data.get(); };
	const unsigned char* data() const { return m_data->data.get(); };
	
	size_t width() const { return m_width; };
	size_t height() const { return m_height; };
	int channelsize() const { return m_channeldepth / 8; };
	int channeldepth() const { return m_channeldepth; };
	int channels() const { return m_channels; };
	int pixelsize() const { return channelsize() * m_channels; };

private:
	size_t m_width;
	size_t m_height;
	int m_channeldepth;
	int m_channels;
	std::shared_ptr<StaticBuffer> m_data;
};

// Adapted from https://gist.github.com/niw/5963798
template<typename Str>
inline Image read_png(const Str& filename) {
	FILE *fp = OPEN_FILE(filename.c_str(), WIDEN("rb"));
	if (!fp) abort();

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png) abort();

	png_infop info = png_create_info_struct(png);
	if (!info) abort();

	if (setjmp(png_jmpbuf(png))) abort();

	png_init_io(png, fp);

	png_read_info(png, info);

	int width = png_get_image_width(png, info);
	int height = png_get_image_height(png, info);
	png_byte color_type = png_get_color_type(png, info);
	png_byte bit_depth = png_get_bit_depth(png, info);

	// Read any color_type into 8bit depth, RGBA format.
	// See http://www.libpng.org/pub/png/libpng-manual.txt

	if (bit_depth == 16)
		png_set_strip_16(png);

	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png);

	// PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png);

	if (png_get_valid(png, info, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png);

	// These color_type don't have an alpha channel then fill it with 0xff.
	if (color_type == PNG_COLOR_TYPE_RGB ||
		color_type == PNG_COLOR_TYPE_GRAY ||
		color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

	if (color_type == PNG_COLOR_TYPE_GRAY ||
		color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		png_set_gray_to_rgb(png);

	png_read_update_info(png, info);

	png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
	for (int y = 0; y < height; y++) 
	{
		row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
	}

	png_read_image(png, row_pointers);

	Image image(width, height, bit_depth, 4);
	unsigned char* data = image.data();
	int pixelsize = image.pixelsize();
	int channelsize = image.channelsize();

	int offset = 0;
	for (int y = 0; y < height; y++) 
	{
		png_bytep row = row_pointers[y];
		for (int x = 0; x < width; x++) 
		{
			png_bytep px = &(row[x * 4]);
			for (int i = 0; i < image.channels(); ++i)
			{
				data[offset + i * channelsize] = px[i];
			}
			offset += pixelsize;
		}
	}

	// Cleanup.
	png_destroy_read_struct(&png, &info, nullptr);
	png = nullptr;
	info = nullptr;
	for (int y = 0; y < height; y++) 
	{
		free(row_pointers[y]);
	}
	free(row_pointers);
	fclose(fp);
	return image;
}

// Adapted from https://gist.github.com/niw/5963798
template<typename Str>
inline void write_png(const Str& filename, const Image& image) {
	FILE *fp = OPEN_FILE(filename.c_str(), WIDEN("wb"));
	if (!fp) abort();

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png) abort();

	png_infop info = png_create_info_struct(png);
	if (!info) abort();

	if (setjmp(png_jmpbuf(png))) abort();

	png_init_io(png, fp);

	// Output is 8bit depth, RGBA format.
	png_set_IHDR(
		png,
		info,
		image.width(), 
		image.height(),
		image.channeldepth(),
		PNG_COLOR_TYPE_RGBA,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT
		);
	png_write_info(png, info);

	// To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
	// Use png_set_filler().
	//png_set_filler(png, 0, PNG_FILLER_AFTER);

	png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * image.height());
	const auto data_ptr = image.data();
	size_t linesize = image.pixelsize() * image.width();

	for (unsigned int i = 0; i < image.height(); ++i)
	{
		row_pointers[i] = (png_bytep)&(data_ptr[i * linesize]);
	}

	png_write_image(png, row_pointers);
	png_write_end(png, NULL);

	// Cleanup.
	png_destroy_write_struct(&png, &info);
	png = nullptr;
	info = nullptr;
	free(row_pointers);
	fclose(fp);
}