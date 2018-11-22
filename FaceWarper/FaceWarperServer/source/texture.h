#pragma once

#include <GL/glew.h>
#include "image.h"

class Texture
{
public:
	Texture(const Image& image);
	~Texture();

	void apply(GLenum unit);

private:
	GLuint m_textureHandle;
};


inline Texture::Texture(const Image& image)
{
	glGenTextures(1, &m_textureHandle);
	glBindTexture(GL_TEXTURE_2D, m_textureHandle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

inline Texture::~Texture()
{
	glDeleteTextures(1, &m_textureHandle);
}

inline void Texture::apply(GLenum unit)
{
	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, m_textureHandle);
}