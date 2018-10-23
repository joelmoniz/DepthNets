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
