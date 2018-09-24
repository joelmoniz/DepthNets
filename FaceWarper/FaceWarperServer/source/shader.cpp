#include "shader.h"

#include <string>
#include <iostream>
#include "io.h"

std::string shader_info_log(GLuint shader_handle)
{
	std::string msg;
	int msg_length = 0;
	glGetShaderiv(shader_handle, GL_INFO_LOG_LENGTH, &msg_length);
	if (msg_length > 1)
	{
		msg.resize(msg_length);
		int written_length;
		glGetShaderInfoLog(shader_handle, msg_length, &written_length, &msg[0]);
		msg.resize(written_length);
	}
	return msg;
}

std::string program_info_log(GLuint shader_handle)
{
	std::string msg;
	int msg_length = 0;
	glGetProgramiv(shader_handle, GL_INFO_LOG_LENGTH, &msg_length);
	if (msg_length > 1)
	{
		msg.resize(msg_length);
		int written_length;
		glGetProgramInfoLog(shader_handle, msg_length, &written_length, &msg[0]);
		msg.resize(written_length);
	}
	return msg;
}

GLuint compile_shader(const ShaderInfo& shader_info)
{
	std::cout << "Compiling shader : " << shader_info.id << "\n";
	
	GLuint shader = glCreateShader(shader_info.type);
	GLchar const* file_pointers[] = { shader_info.data.c_str() };
	glShaderSource(shader, 1, file_pointers, NULL);

	glCompileShader(shader);
	std::string info_msg = shader_info_log(shader);
	if (info_msg.size() == 0)
	{
		std::cout << "No errors." << std::endl;
	}
	else
	{
		std::cout << "Errors : \n" << info_msg << std::endl;
	}
	return shader;
}

GLuint link_program(const std::vector<GLuint>& shaders)
{
	std::cout << "Linking program\n";

	GLuint program = glCreateProgram();
	for (auto s : shaders)
	{
		glAttachShader(program, s);
	}
	glLinkProgram(program);
	std::string info_msg = program_info_log(program);
	if (info_msg.size() == 0)
	{
		std::cout << "No errors." << std::endl;
	}
	else
	{
		std::cout << "Errors : \n" << info_msg << std::endl;
	}
	return program;
}

void ShaderProgram::compile()
{
	std::vector<GLuint> shaders;
	for (auto& sinfo : m_shader_info_list)
	{
		shaders.push_back(compile_shader(sinfo));
	}
	m_program_handle = link_program(shaders);
	m_is_compiled = true;
}

void ShaderProgram::enable() const
{
	glUseProgram(m_program_handle);
}


const char* DEFAULT_VERTEX_SHADER_STRING = R"(
#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 in_texcoord;

uniform mat4 projection;
out vec2 texcoord;

void main()
{
    gl_Position = projection * vec4(position, 1.0);
    texcoord = in_texcoord;
}
)";

const char* DEFAULT_FRAGMENT_SHADER_STRING = R"(
#version 330

uniform sampler2D tex;
in vec2 texcoord;
out vec4 color;

void main (void) 
{
	color = texture(tex, texcoord);
}
)";