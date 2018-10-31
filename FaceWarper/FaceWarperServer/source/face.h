#pragma once

#include <array>
#include <glm/glm.hpp>
#include "geo.h"
#include "texture.h"

typedef std::vector<std::array<int, 3>> TopoContainer;

extern const TopoContainer DEFAULT_FACE_TOPOLOGY_RAW;

inline TopoContainer to_zero_index(const TopoContainer& indices)
{
	TopoContainer zero_indices;
	zero_indices.reserve(indices.size());
	for (const auto& triple : indices)
	{
		zero_indices.emplace_back(decltype(zero_indices)::value_type{ triple[0] - 1,triple[1] - 1,triple[2] - 1 });
	}
	return zero_indices;
}

class FaceTopology
{	
public:
	FaceTopology() {};
	FaceTopology(const TopoContainer& topo) : m_topo(topo) {};

	std::vector<unsigned int> indexBuffer() const
	{
		std::vector<unsigned int> indices;
		indices.reserve(m_topo.size() * 3);
		for (const auto& t : m_topo)
		{
			indices.push_back(t[0]);
			indices.push_back(t[1]);
			indices.push_back(t[2]);
		}
		return indices;
	}

private:
	TopoContainer m_topo;
};

struct FaceKeypoint
{
	FaceKeypoint() : u(0.0), v(0.0), depth(0.0) {};
	FaceKeypoint(float u, float v, float depth) : u(u), v(v), depth(depth) {};
	float u, v, depth;
};

struct FaceVertex
{
	FaceVertex(float x, float y, float z) : x(x), y(y), z(z) {};
	float x, y, z;
};

class IFaceTransform
{
public:
	virtual FaceVertex apply(const FaceVertex& v) const = 0;
};

class AffineTransform : public IFaceTransform
{
public:
	AffineTransform(const glm::mat4& transform) : m_transform(transform) {};
	virtual ~AffineTransform() = default;

	virtual FaceVertex apply(const FaceVertex& v) const override
	{
		glm::vec4 point(v.x, v.y, v.z, 1.0);
		point = m_transform * point;
		return FaceVertex(point.x, point.y, point.z);
	}

private:
	glm::mat4 m_transform;
};

typedef std::vector<FaceKeypoint> FaceKeypointContainer;

class Face
{
public:
	typedef std::vector<Vertex> VertexContainer;
	typedef std::vector<unsigned int> IndexContainer;

public:
	Face(const FaceTopology& topology, const FaceKeypointContainer& keypoints, const Image& image, const IFaceTransform& transform)
		: m_indices(topology.indexBuffer()),
		m_image(image)
	{
		m_vertices.reserve(keypoints.size());
		const float width = static_cast<float>(m_image.width());
		const float height = static_cast<float>(m_image.height());
		const float iWidth = 1.0f / width;
		const float iHeight = 1.0f / height;
		for (const auto& kp : keypoints)
		{
			m_vertices.emplace_back();
			auto& v = m_vertices.back();
			v.Texture[0] = kp.u;
			v.Texture[1] = kp.v;

			FaceVertex fv = transform.apply({ kp.u * width, kp.v * height, kp.depth });

			v.Position.x = fv.x * iWidth;
			v.Position.y = fv.y * iHeight;
			v.Position.z = fv.z;
		}

		initGlBuffers();
		m_texture = std::make_shared<Texture>(m_image);
	}

	~Face()
	{
		deleteGlBuffers();
	}

	void draw();
	const Image& image() const { return m_image; };

private:
	void initGlBuffers();
	void deleteGlBuffers();

private:
	VertexContainer m_vertices;
	IndexContainer m_indices;

	GLuint m_vertexBuffer;
	GLuint m_vertexArray;
	GLuint m_indexBuffer;

	Image m_image;
	std::shared_ptr<Texture> m_texture;
};

inline void Face::initGlBuffers()
{
	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(VertexContainer::value_type), m_vertices.data(), GL_STATIC_DRAW);

	glGenVertexArrays(1, &m_vertexArray);
	glBindVertexArray(m_vertexArray);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);

	glGenBuffers(1, &m_indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(IndexContainer::value_type), m_indices.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	// Buffer format :
	// - texture coord (2 float).
	// - normal (3 float).
	// - position (3 float).
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,                  // attribute 0
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		sizeof(VertexContainer::value_type),     // stride
		(void*)(sizeof(float) * 2 + sizeof(Vertex::Normal))  // array buffer offset.
														//(void*)0
		);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1,                  // attribute 1
		2,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		sizeof(VertexContainer::value_type),     // stride
		(void*)(0)  // array buffer offset.
		);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}

inline void Face::deleteGlBuffers()
{
	glDeleteBuffers(1, &m_vertexBuffer);
	glDeleteBuffers(1, &m_indexBuffer);
	glDeleteVertexArrays(1, &m_vertexArray);
}

inline void Face::draw()
{
	if (!m_vertices.empty() && !m_indices.empty())
	{
		const GLenum textureUnit = 0;
		m_texture->apply(textureUnit);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
		glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, (void*)0);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}
}