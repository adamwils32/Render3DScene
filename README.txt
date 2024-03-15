OpenGL Textured Mesh Renderer README

Overview:
This Python project demonstrates a simple OpenGL application capable of rendering 3D meshes with textures. It leverages
PyOpenGL for interfacing with the OpenGL API, OpenCV for texture loading, and NumPy for efficient mathematical
operations. The primary objective is to illustrate fundamental concepts of 3D graphics programming, such as handling
vertices and textures, shader programming, and camera manipulation.

Notes for User(s):
- In main.py, .ply and .bmp files are assumed to be stored in a folder called RenderSceneAssets for cleaner project
structure. This is why the assignment was submitted as a .zip instead of just submitting the relevant files.

OpenGL Setup:
The application initializes an OpenGL context using GLFW, sets up depth testing for accurate z-ordering in the rendering
process, and enables blending to support texture transparency.

Known Bugs:
- Transparent Object Sorting: The application does not perform back-to-front sorting of transparent objects before
rendering. Instead, objects are hard coded to be processed in the correct order.
- Fixed Aspect Ratio: The perspective projection matrix is calculated using a fixed aspect ratio. If the window is
resized, the aspect ratio might not match, potentially distorting the rendered scene.

Running the Application:

Requirements:
- Python 3.x
- PyOpenGL
- GLFW (Python bindings)
- OpenCV-Python
- NumPy
- plyfile (for PLY file handling)

Environment Setup:
1. Ensure Python 3.x is installed.
2. Install the required Python packages with the following command:
   pip install PyOpenGL glfw opencv-python numpy plyfile

Execution:
1. Navigate to the directory containing main.py and textured_mesh.py.
2. Execute the application:
   python main.py
3. Control the camera using the UP and DOWN arrow keys to move forward and backward, and the LEFT and RIGHT arrow keys
   to rotate.

Screenshot 1 Steps:
1. Turn left roughly 90 degrees until the pitchfork is centered in the window

Screenshot 2 Steps:
1. Turn right until the sink's drain is in the middle of the screen
2. Move forward (towards the sink) until only the mirror above the sink is visible, the sink should be below your FOV
3. Turn left until the table with the apples on it is in the middle of the screen
