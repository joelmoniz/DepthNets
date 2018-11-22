#pragma once

#include <vector>
#include <string>
#include <GL/glew.h>

extern const char* DEFAULT_VERTEX_SHADER_STRING;
extern const char* DEFAULT_FRAGMENT_SHADER_STRING;

struct ShaderInfo
{
	ShaderInfo(const std::string& id, const std::string& data, unsigned int type)
		: id(id), data(data), type(type) {}

	std::string id;
	std::string data;
	unsigned int type;
};

class ShaderProgram
{
public:
	ShaderProgram(std::vector<ShaderInfo> shader_info_list)
		: m_shader_info_list(std::move(shader_info_list)),
		m_is_compiled(false),
		m_program_handle(0)
	{}

	GLuint get_handle() const { return m_program_handle; }
	void enable() const;

	void compile();

private:
	std::vector<ShaderInfo> m_shader_info_list;
	bool m_is_compiled;
	GLuint m_program_handle;
};
