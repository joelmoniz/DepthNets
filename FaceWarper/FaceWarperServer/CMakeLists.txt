cmake_minimum_required (VERSION 3.9)
project(FaceWarperServer)
include_directories(source)
file(GLOB SOURCES "source/*.cpp" "source/*.h")
add_executable(FaceWarperServer ${SOURCES})

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
target_link_libraries(FaceWarperServer ${PNG_LIBRARIES})

find_package(OpenGL)
target_link_libraries(FaceWarperServer ${OPENGL_gl_LIBRARY})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
target_link_libraries(FaceWarperServer ${GLUT_LIBRARIES})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
target_link_libraries(FaceWarperServer ${GLEW_LIBRARIES})

find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

