#pragma once

#include <glm/glm.hpp>

struct Vertex
{
	float Texture[2];
	glm::vec3 Normal;
	glm::vec3 Position;


	Vertex()
		: Normal(0.0f)
		, Position(0.0f)
	{
		Texture[0] = 0.0f;
		Texture[1] = 0.0f;
	}

	Vertex(const glm::vec3& P, const glm::vec3& N = glm::vec3(0.0f))
		: Normal(N)
		, Position(P)
	{
		Texture[0] = 0.0f;
		Texture[1] = 0.0f;
	}
};