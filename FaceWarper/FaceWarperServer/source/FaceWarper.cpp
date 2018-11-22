#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include "fbo.h"
#include "io.h"
#include "platform.h"

///////////////////////////////////////////////////////////////////////////////
// Global variables
ShaderProgram faceShader({ {"vertex", DEFAULT_VERTEX_SHADER_STRING, GL_VERTEX_SHADER}, {"fragment", DEFAULT_FRAGMENT_SHADER_STRING, GL_FRAGMENT_SHADER} });
FaceTopology faceTopology;
std::shared_ptr<FBO> offscreenFBO;
const std::string READY("ready");
const float FRAMERATE = 10000000.0f;

struct WindowInfo
{
	WindowInfo()
		: currentW(360), currentH(360), g_nWindowID(0) {}

	int currentW;
	int currentH;
	int g_nWindowID;
};
WindowInfo global_WindowInfo;

///////////////////////////////////////////////////////////////////////////////
// Forward declarations
void init();
void drawFace(Face& face, bool invertHeight);
void draw();
void idle();
void keyboard(unsigned char key, int x, int y);
void keyboardSpecial(int key, int x, int y);
void mouseMove(int x, int y);
void mouseClick(int button, int state, int x, int y);
void resizeWindow(GLsizei w, GLsizei h);
void compileShaders();

int main(int argc,char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH); // | GLUT_DOUBLE
	glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_CORE_PROFILE);
    glutInitWindowSize(global_WindowInfo.currentW, global_WindowInfo.currentH);
	global_WindowInfo.g_nWindowID = glutCreateWindow("Face warper");

    glutDisplayFunc(draw);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(keyboardSpecial);
    glutReshapeFunc (resizeWindow);
    glutMotionFunc (mouseMove);
    glutMouseFunc (mouseClick);

	// glewExperimental necessary for GLEW with OpenGL >=3.2
	glewExperimental = true;
	GLenum err = glewInit();
	if (!glewIsSupported("GL_VERSION_3_3"))
	{
		std::cout << "OpenGL 3.3 not supported.\n";
		exit(1);
    }
	std::cout << "OpenGL version = " << glGetString(GL_VERSION) << std::endl;

	compileShaders();
    init();

    glutMainLoop();

    return EXIT_SUCCESS;
}

void compileShaders()
{
	faceShader.compile();
}

void init() {
    glClearColor (0.0, 0.0, 0.0, 1.0);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	faceTopology = to_zero_index(DEFAULT_FACE_TOPOLOGY_RAW);
	offscreenFBO = std::make_shared<FBO>();
}


void drawFace(Face& face, bool invertHeight)
{
	glDisable(GL_CULL_FACE);

	faceShader.enable();
	glm::mat4 view;
	if (invertHeight)
	{
		view = glm::ortho(0.0, 1.0, 1.0, 0.0, -100000.0, 100000.0);
	}
	else
	{
		view = glm::ortho(0.0, 1.0, 0.0, 1.0, -100000.0, 100000.0);
	}
	const auto location = glGetUniformLocation(faceShader.get_handle(), "projection");
	glUniformMatrix4fv(location, 1, false, glm::value_ptr(view));
	const GLenum textureUnit = 0;
	const auto textureLocation = glGetUniformLocation(faceShader.get_handle(), "tex");
	glUniform1i(textureLocation, textureUnit);
	face.draw();

    glFlush();
}

struct InputFiles
{
	filepath_string image;
	filepath_string depth;
	filepath_string keypoints;
	filepath_string affine;
	filepath_string transformedImage;
};

InputFiles read_filepaths()
{
	InputFiles files;
	standard_input_stream >> files.image >> files.keypoints >> files.depth >> files.affine >> files.transformedImage;
	return files;
}

void draw(void)
{ 
	std::cout << READY << std::endl;

	InputFiles files = read_filepaths();

	Image image = read_png(files.image);
	if (offscreenFBO->dimensions() != Dimensions(image.width(), image.height()))
	{
		offscreenFBO->init(image.width(), image.height());
	}
	AffineTransform transform = read_affine_transform(files.affine);
	FaceKeypointContainer keypoints = read_face_keypoints(files.keypoints, files.depth, image.width(), image.height());
	Face face(faceTopology, keypoints, image, transform);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Current face visualization in display window.
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glViewport(0, 0, global_WindowInfo.currentW, global_WindowInfo.currentH);
	drawFace(face, true);

	// Render face in offscreen buffer.
	offscreenFBO->enable();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	const size_t width = face.image().width();
	const size_t height = face.image().height();
	glViewport(0, 0, width, height);
	drawFace(face, false);

	// Save offscreen buffer to png file.
	Image tmpImage(width, height, face.image().channeldepth(), face.image().channels());
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, tmpImage.data());
	write_png(files.transformedImage, tmpImage);
	offscreenFBO->disable();

    // No call to glutSwapBuffers() since we don't need to be in sync with vsync (unnecessary slowdown).
}

void idle(void)
{
    static int nWaitUntil = glutGet(GLUT_ELAPSED_TIME);

    int nTimer = glutGet(GLUT_ELAPSED_TIME);
    if(nTimer >= nWaitUntil)
    {
		glutPostRedisplay();
		nWaitUntil = nTimer + (int)(1000.0f / FRAMERATE);
    }
}

void keyboard(unsigned char key, int x, int y)
{
   
}

void keyboardSpecial(int touche, int x, int y)
{
    
}

void resizeWindow(GLsizei w, GLsizei h)
{
	glutReshapeWindow(global_WindowInfo.currentW, global_WindowInfo.currentH);
}

void mouseClick(int button, int state, int x, int y)
{

}

void mouseMove(int x, int y)
{
   
}

const TopoContainer DEFAULT_FACE_TOPOLOGY_RAW = { {1,37,18},{1,2,37},{2,42,37},{2,3,42},{3,32,42},{3,50,32},{3,4,50},{4,49,50},{4,5,49},{5,6,49},{6,7,49},{7,60,49},{7,8,60},{8,59,60},{8,9,59},{9,58,59},{9,57,58},{9,10,57},{10,56,57},{10,11,56},{11,12,56},{12,55,56},{12,13,55},{13,14,55},{14,54,55},{14,15,54},{15,36,54},{15,47,36},{15,16,47},{16,46,47},{16,17,46},{17,27,46},{27,26,46},{26,45,46},{26,25,45},{25,44,45},{25,24,44},{24,43,44},{24,23,43},{23,28,43},{23,22,28},{22,40,28},{22,21,40},{21,39,40},{21,20,39},{20,38,39},{20,19,38},{19,37,38},{19,18,37},{37,42,38},{38,42,39},{39,42,41},{39,41,40},{43,48,44},{44,48,47},{44,47,45},{45,47,46},{49,60,61},{61,60,68},{60,59,68},{68,59,67},{59,58,67},{58,57,67},{57,66,67},{57,56,66},{56,65,66},{56,55,65},{55,54,65},{54,64,65},{54,53,64},{53,63,64},{53,52,63},{52,51,63},{51,62,63},{51,50,62},{50,61,62},{50,49,61},{61,68,62},{62,68,63},{63,68,67},{63,67,66},{63,66,64},{64,66,65},{40,29,28},{40,30,29},{40,41,30},{41,32,30},{41,42,32},{30,32,31},{31,32,33},{31,33,34},{31,34,35},{31,35,36},{31,36,30},{36,47,48},{30,36,48},{30,48,43},{30,43,29},{29,43,28},{32,50,51},{32,51,33},{33,51,52},{33,52,34},{34,52,35},{35,52,53},{36,35,53},{36,53,54} };
