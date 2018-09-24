@echo off
if not exist .\build\ mkdir .\build
cd .\build\
cmake -DCMAKE_PREFIX_PATH=".\lib\zlib;.\lib\libpng;.\lib\freeglut;.\lib\glm;.\lib\glew" ..