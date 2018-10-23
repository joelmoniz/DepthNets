# FaceWarper

FaceWarper is a program that applies a 3D affine transform to an image and saves the result to disk. It uses a client-server architecture. The server, FaceWarperServer, is programmed in C++11. A reference Python implementation of a client, FaceWarperClient, is provided. Finaly, the Python script "warp_dataset.py" is provided to apply on a full dataset the 3D affine transforms infered using DepthNet.

## Usage with DepthNet
You first need to build FaceWarperServer in order to use it. Detailed instructions on how to build it is provided in a section below.

## FaceWarperServer
FaceWarperServer is a C++11 application that applies a 3D affine transform to a mesh. It acts as a server that listens on the standard input. A visualization window is shown on the desktop. The image in this window is scaled to ease visualization, but the processing is done at the source texture resolution.

The server supports Linux and Windows.

### Dependencies
To build FaceWarperServer, the following libraries are required :
- GLEW
- freeglut
- GLM
- libpng (requires zlib)

On a Debian/Ubuntu Linux distribution, these dependencies can be installed with the following packages :
- libglew-dev
- freeglut3-dev
- libglm-dev
- libpng-dev

### Build
A CMake CMakeLists.txt is provided to build FaceWarperServer. To build on Linux, the helper shell script "build_linux.sh" is provided. To build on Windows, the helper Batch script "build_windows.bat" is provided. It supposes that the required libraries are located in "FaceWarperServer\lib" directory.

If you have problems building with CMake on Windows, a Visual Studio 2017 project is also provided in the "windows_build" directory.

If one of the dependencies can't be found by CMake, use the "-DCMAKE_PREFIX_PATH=" option to tell CMake where to find them. For example, if CMake can't find GLEW :
```
cmake -DCMAKE_PREFIX_PATH="/path/to/glew/glew-2.1.0/build/" ..
```

## warp_dataset.py
The Python program "warp_dataset.py" launches FaceWarperServer and sequentially sends each face in the dataset to be warped.

## Implementation details

### FaceWarperClient
FaceWarperClient is a Python package that can manage a FaceWarperServer and send warp commands to the server. It is used by "warp_dataset.py" to communicate with the FaceWarperServer. The package can be useful if you want to write a custom script to send commands to a FaceWarperServer instead of using the supplied "warp_dataset.py".

### Warp command format
This section presents the command format used to send warp commands to FaceWarperServer. These are implementation details that are not necessary to understand if you just want to use the supplied FaceWarperClient package or the supplied execution script 'warp_dataset.py'. It is useful is you want to implement a new client.

The FaceWarperServer listens on the standard input for warping commands. Each warping command is a space delimited line containing the following information, in order:

1. source image file path
2. source 2D keypoint file path
3. source keypoint depth file path
4. affine transform file path
5. output image path

Once the server has finished processing the command, it writes "ready" on a new line on the standard output to notify the client that it is ready to receive a new warping command.
