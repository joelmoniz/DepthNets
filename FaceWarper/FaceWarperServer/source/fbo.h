#pragma once

#include <iostream>
#include <GL/glew.h>

struct Dimensions
{
	Dimensions() : width(0), height(0) {}
	Dimensions(int width, int height) : width(width), height(height) {}

	int width;
	int height;
};

inline bool operator==(const Dimensions& left, const Dimensions& right) { return left.width == right.width && left.height == right.height; }
inline bool operator!=(const Dimensions& left, const Dimensions& right) { return !(left == right); }

class FBO
{
public:
	FBO();
	~FBO();

	void init(int width, int height);
	void enable() const;
	void disable() const;
	Dimensions dimensions() const { return Dimensions(m_width, m_height); }

private:
	void free();

private:
	GLuint m_fbo_handle;
	GLuint m_color_texture_handle;
	GLuint m_depth_texture_handle;
	GLint  m_internal_format;
	GLenum m_texturing_target;

	int m_height;
	int m_width;
};

inline FBO::FBO()
	:	m_fbo_handle(0),
		m_color_texture_handle(0),
		m_depth_texture_handle(0),
		m_internal_format(0),
		m_texturing_target(GL_TEXTURE_2D),
		m_height(0),
		m_width(0)
{
	
}

inline FBO::~FBO()
{
	this->free();
}

inline void FBO::init(int width, int height)
{
	this->free();

	m_width = width;
	m_height = height;

	glGenFramebuffers(1, &m_fbo_handle);
	glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_handle);

	// Initialize color texture.
	glGenTextures(1, &m_color_texture_handle);
	glBindTexture(m_texturing_target, m_color_texture_handle);
	glTexParameteri(m_texturing_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(m_texturing_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(m_texturing_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(m_texturing_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(m_texturing_target, 0, GL_RGB, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_texturing_target, m_color_texture_handle, 0);

	// Initialize depth texture.
	glGenTextures(1, &m_depth_texture_handle);
	glBindTexture(m_texturing_target, m_depth_texture_handle);
	glTexParameteri(m_texturing_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(m_texturing_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(m_texturing_target, 0, GL_DEPTH_COMPONENT, m_width, m_height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_texturing_target, m_depth_texture_handle, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE:
		break; 
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
		std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT" << std::endl;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		std::cerr << "GL_FRAMEBUFFER_UNSUPPORTED" << std::endl;
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
		std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT" << std::endl;
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS" << std::endl;
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_FORMATS" << std::endl;
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
		std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER" << std::endl;
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
		std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT" << std::endl;
		break;
	default:
		std::cerr << "UNKNOWN ERROR" << std::endl;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline void FBO::enable() const
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_handle);
}

inline void FBO::disable() const
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline void FBO::free()
{
	if (m_color_texture_handle)
	{
		glDeleteTextures(1, &m_color_texture_handle);
		m_color_texture_handle = 0;
	}

	if (m_depth_texture_handle)
	{
		glDeleteTextures(1, &m_depth_texture_handle);
		m_depth_texture_handle = 0;
	}

	if (m_fbo_handle)
	{
		glDeleteFramebuffers(1, &m_fbo_handle);
		m_fbo_handle = 0;
	}
}