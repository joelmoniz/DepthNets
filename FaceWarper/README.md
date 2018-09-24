# FaceWarper

## FaceWarperServer
FaceWarperServer is a C++11 application that applies a 3D affine transform to a mesh. It acts as a server that listens on the standard input. A visualization window is shown on the desktop. The image in this window is scaled to ease visualization, but the processing is done at the source texture resolution.

The server supports Linux and Windows.

### Dependencies

- GLEW
- freeglut
- GLM
- libpng (requires zlib)

### Build
A CMake CMakeLists.txt is provided to build FaceWarperServer. To build on Linux, the helper shell script "build_linux.sh" is provided. To build on Windows, the helper Batch script "build_windows.bat" is provided. It supposes that the required libraries are located in "FaceWarperServer\lib" directory.

If you have problems building with CMake on Windows, a Visual Studio 2017 project is also provided in the "windows_build" directory.

## FaceWarperClient
FaceWarperClient is a Python package that can manage a FaceWarperServer and send warp commands to the server.

## warp_dataset.py
The Python program "warp_dataset.py" launches FaceWarperServer and sequentially sends each face in the dataset to be warped.

## Warp command format

Each warping command is a space delimited line containing the following information, in order:

1. source texture file path
2. keypoint file path
3. depth file path
4. affine transform file path
5. destination image file path
