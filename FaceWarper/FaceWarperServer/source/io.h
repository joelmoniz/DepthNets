#pragma once

#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include "face.h"
#include "platform.h"

inline FaceKeypointContainer read_face_keypoints(const filepath_string& positionFilepath, const filepath_string& depthFilepath, int width, int height)
{
	std::ifstream positionFile;
	positionFile.open(positionFilepath);
	std::ifstream depthFile;
	depthFile.open(depthFilepath);

	FaceKeypointContainer keypoints;

	while (true)
	{
		float u, v, depth;
		positionFile >> u >> v;
		depthFile >> depth;

		if (positionFile.eof() || depthFile.eof()) break;

		u = u / width;
		v = v / height;
		keypoints.emplace_back(u, v, depth);
	}
	return keypoints;
}

inline AffineTransform read_affine_transform(const filepath_string& filepath)
{
	std::ifstream file;
	file.open(filepath);
	
	glm::mat4 matrix(0.0f);
	int line = 0;
	while (true)
	{
		float a, b, c, d;
		file >> a >> b >> c >> d;

		if (file.eof()) break;

		matrix[line] = glm::vec4(a, b, c, d);

		++line;
	}

	matrix[2] = glm::vec4(0.0);
	matrix[3] = glm::vec4(0.0);

	return AffineTransform(glm::transpose(matrix));
}