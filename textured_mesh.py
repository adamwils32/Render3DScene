"""
File: textured_mesh.py

Description:
This module is part of the OpenGL Textured Mesh Renderer project. It defines the structure and functionality
necessary for creating, loading, and rendering textured meshes within a 3D environment. The primary focus is on
defining mesh vertices, loading mesh data from PLY files, handling texture images from BMP files, and integrating
these elements through OpenGL to render the textured meshes effectively.

Key Features:
- VertexData and TriData classes for organizing the geometric data of meshes.
- A function for reading mesh data from PLY format files, facilitating the use of complex 3D models.
- Implementation of texture loading using OpenCV, supporting image processing to match OpenGL's requirements.
- Setup and management of OpenGL data structures (VAOs, VBOs, EBOs) for rendering.
- Shader program initialization for applying textures and rendering the meshes with basic lighting.

Usage:
This module is intended to be used in conjunction with 'main.py', which serves as the entry point for initiating
the OpenGL context, setting up the rendering loop, and handling user inputs. The 'TexturedMesh' class within this
module is instantiated in 'main.py' to create and render individual mesh objects.

Requirements:
- OpenGL (via PyOpenGL)
- OpenCV-Python for texture image processing
- numpy for efficient mathematical operations
- plyfile for parsing PLY files

Author: Adam Wilson
Date: March 8, 2024
"""

from OpenGL.GL import *
import numpy as np
import cv2
from plyfile import PlyData

# Define a class to store vertex data including position, normal, and texture coordinates.
class VertexData:
    def __init__(self, x, y, z, nx=0, ny=0, nz=0, u=0, v=0):
        # Initialize vertex position, normal, and texture coordinates (u, v).
        self.position = (x, y, z)  # Vertex position in 3D space.
        self.normal = (nx, ny, nz)  # Vertex normal for lighting calculations.
        self.texCoords = (u, v)  # Texture coordinates for texture mapping.


# Define a class to store triangle data, specifically the indices of the vertices that form a triangle.
class TriData:
    def __init__(self, v1, v2, v3):
        # Initialize triangle with vertex indices.
        self.indices = (v1, v2, v3) # Indices into the vertex array forming the triangle.


# Function to read vertex and triangle data from a PLY file.
def readPLYFile(fname):
    vertices = []
    faces = []
    plydata = PlyData.read(fname)

    # Iterate through vertex elements in the PLY file to extract position, normal, and texture coordinates.
    for vertex in plydata['vertex']:
        vertices.append(VertexData(vertex['x'], vertex['y'], vertex['z'],
                                   vertex['nx'], vertex['ny'], vertex['nz'],
                                   vertex['u'], vertex['v']))

    # Iterate through face elements to extract vertex indices forming each triangle.
    for face in plydata['face']:
        faces.append(TriData(*face['vertex_indices']))

    return vertices, faces  # Return lists of vertices and triangles.


# Class to represent a textured mesh, including its geometry (vertices and triangles) and associated texture.
class TexturedMesh:
    def __init__(self, plyFilePath, bmpFilePath):
        # Load vertex and triangle data from a PLY file and texture from a BMP file.
        self.vertices, self.faces = readPLYFile(plyFilePath)
        self.textureID = self.loadTexture(bmpFilePath)
        self.shaderProgram = self.initializeShaderProgram()
        self.setupMesh()

    # Set up vertex array object (VAO), vertex buffer object (VBO), and element buffer object (EBO) for rendering.
    def setupMesh(self):
        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        # Bind the VAO to store the following VBO and attribute configurations.
        glBindVertexArray(self.VAO)
        # Bind and fill VBO with vertex data.
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertexData(), GL_STATIC_DRAW)

        # Bind and fill EBO with index data for drawing triangles.
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indexData(), GL_STATIC_DRAW)

        # Specify the layout of vertex data in the VBO.
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)  # Enable vertex position attribute.

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)  # Enable vertex normal attribute.

        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)  # Enable texture coordinate attribute.

        # Unbind the VBO and VAO now that we're done setting up.
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    # Prepare vertex data for OpenGL: combines position, normal, and texture coordinates of each vertex into a single array.
    def vertexData(self):
        data = []

        # Iterate through each vertex and extend the data list with vertex attributes.
        for vertex in self.vertices:
            data.extend([vertex.position[0], vertex.position[1], vertex.position[2],
                         vertex.normal[0], vertex.normal[1], vertex.normal[2],
                         vertex.texCoords[0], vertex.texCoords[1]])
        return np.array(data, dtype=np.float32)  # Convert to NumPy array for OpenGL.

    # Prepare index data for OpenGL: creates an array of indices for drawing triangles.
    def indexData(self):
        indices = []
        # Iterate through each triangle and extend the indices list with the triangle's vertex indices.
        for face in self.faces:
            indices.extend(face.indices)
        return np.array(indices, dtype=np.uint32)  # Convert to NumPy array for OpenGL.

    # Load a texture from a BMP file using OpenCV, convert to an appropriate format, and create an OpenGL texture.
    def loadTexture(self, filePath):

        # Use OpenCV to load the image, with consideration for alpha channels.
        image = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise Exception(f"Failed to load texture: {filePath}")

        # Convert image to RGBA format depending on the original format.
        if image.shape[2] == 4:  # Image already has an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:  # No alpha channel, convert BGR to RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        elif image.shape[2] == 1:  # Grayscale to RGBA
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

        # Flip the image vertically to match OpenGL's texture coordinate system.
        image = cv2.flip(image, 0)

        # Flatten the image to a 1D array for OpenGL texture creation.
        img_data = image.flatten()
        img_width, img_height = image.shape[1], image.shape[0]

        # Generate and configure an OpenGL texture.
        textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, textureID)

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Upload the texture data to GPU
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        # Generate mipmaps
        glGenerateMipmap(GL_TEXTURE_2D)

        # Unbind the texture
        glBindTexture(GL_TEXTURE_2D, 0)

        return textureID

    # Initialize a shader program for rendering the textured mesh.
    def initializeShaderProgram(self):
        # Define vertex and fragment shader source code.
        # Vertex shader processes each vertex's position and texture coordinates.
        vertexShaderSource = """
        #version 330 core
        // Input vertex data, different for all executions of this shader.
        layout(location = 0) in vec3 vertexPosition;
        layout(location = 2) in vec2 uv;
        // Output data ; will be interpolated for each fragment.
        out vec2 uv_out;
        // Values that stay constant for the whole mesh.
        uniform mat4 MVP;
        void main(){
        // Output position of the vertex, in clip space : MVP * position
            gl_Position =  MVP * vec4(vertexPosition,1);
            // The color will be interpolated to produce the color of each fragment
            uv_out = uv;
        }"""

        # Fragment shader samples from the texture and outputs the final color.
        fragmentShaderSource = """
        #version 330 core
        in vec2 uv_out;
        uniform sampler2D tex;
        out vec4 FragColor;
        void main() {
            FragColor = texture(tex, uv_out);
        }
        """

        # Compile the vertex and fragment shaders.
        vertexShader = self.compileShader(GL_VERTEX_SHADER, vertexShaderSource)
        fragmentShader = self.compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource)

        # Create a shader program and link the shaders.
        shaderProgram = glCreateProgram()
        glAttachShader(shaderProgram, vertexShader)
        glAttachShader(shaderProgram, fragmentShader)
        glLinkProgram(shaderProgram)

        # Check for errors in linking the shader program.
        if glGetProgramiv(shaderProgram, GL_LINK_STATUS) != GL_TRUE:
            infoLog = glGetProgramInfoLog(shaderProgram)
            raise RuntimeError(f'Error linking shader program: {infoLog}')

        # Cleanup: shaders can be deleted once linked into the program.
        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)

        return shaderProgram

    # Compiles a shader from source code.
    def compileShader(self, shaderType, source):
        shader = glCreateShader(shaderType)  # Create a shader object.
        glShaderSource(shader, source)  # Set the shader source code.
        glCompileShader(shader)  # Compile the shader.

        # Check for errors in shader compilation.
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            infoLog = glGetShaderInfoLog(shader)
            raise RuntimeError(f'Error compiling shader: {infoLog}')

        return shader  # Return the compiled shader ID.

    # Renders the mesh using the shader program and bound texture.
    def draw(self, MVP):
        glUseProgram(self.shaderProgram)  # Use the shader program for rendering.
        glUniformMatrix4fv(glGetUniformLocation(self.shaderProgram, "MVP"), 1, GL_FALSE, MVP)  # Set the MVP uniform.

        # Bind the texture to texture unit 0 and set the 'tex' uniform.
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glUniform1i(glGetUniformLocation(self.shaderProgram, "tex"), 0)

        # Bind the VAO and draw the mesh using the element buffer object (EBO).
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.faces) * 3, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)  # Unbind the VAO.
        glUseProgram(0)  # Unuse the shader program.
