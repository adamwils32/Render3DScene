"""
File: main.py

Description:
This script serves as the entry point for the OpenGL Textured Mesh Renderer project. It initializes the OpenGL context,
sets up the window using GLFW, and contains the main rendering loop. The script demonstrates basic 3D graphics
programming concepts in Python, including handling vertex and texture data, shader programming, and camera manipulation.
It enables user interaction for navigating the 3D scene with keyboard inputs.

Usage: Execute `python main.py` from the terminal to run the application.

Requirements: PyOpenGL, GLFW for Python, OpenCV-Python, numpy, plyfile

Author: Adam Wilson
Date: March 8, 2024
"""

import glfw
from OpenGL.GL import *
import numpy as np
from textured_mesh import TexturedMesh

# Define initial camera settings, including position, direction it's facing, and 'up' vector for orientation.
camera_pos = np.array([0.5, 0.4, 0.5], dtype=np.float32)
camera_front = np.array([0, 0, -1], dtype=np.float32)
camera_up = np.array([0, 1, 0], dtype=np.float32)
camera_speed = 2.5  # Adjust as needed

# Variables for tracking time between frames to calculate smooth movement.
delta_time = 0.0  # Time between current frame and last frame
last_frame = 0.0  # Time of last frame

# Global variable for the horizontal angle towards which the camera points.
yaw = -90.0  # Initialized facing towards negative Z


def process_input(window):  # Process keyboard input to move the camera or adjust its direction.
    global camera_pos, camera_front, camera_speed, delta_time, yaw

    # Move the camera forward or backward.
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera_pos += camera_speed * delta_time * camera_front
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera_pos -= camera_speed * delta_time * camera_front

    camera_speed_rotation = 50  # Adjust rotation speed as needed
    # Rotate the camera left or right.
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        yaw -= camera_speed_rotation * delta_time
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        yaw += camera_speed_rotation * delta_time

    # Recalculate the direction the camera is facing.
    front = np.array([
        np.cos(np.radians(yaw)) * np.cos(np.radians(0)),  # Assume pitch=0 for simplicity
        np.sin(np.radians(0)),  # Assume pitch=0 for simplicity
        np.sin(np.radians(yaw)) * np.cos(np.radians(0))  # Assume pitch=0 for simplicity
    ])

    camera_front = front / np.linalg.norm(front)


def lookAt(eye, center, up):
    # Generates a view matrix representing the camera's orientation and position in the world.
    f = np.array(center - eye)
    f = f / np.linalg.norm(f)
    u = np.array(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    M = np.identity(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -np.array([np.dot(s, eye), np.dot(u, eye), np.dot(-f, eye)])
    return M


def perspective(fov, aspect, zNear, zFar):
    # Creates a perspective projection matrix based on field of view, aspect ratio, and near/far planes.
    tanHalfFovy = np.tan(fov / 2)
    M = np.zeros((4, 4))
    M[0, 0] = 1 / (aspect * tanHalfFovy)
    M[1, 1] = 1 / (tanHalfFovy)
    M[2, 2] = -(zFar + zNear) / (zFar - zNear)
    M[2, 3] = -(2 * zFar * zNear) / (zFar - zNear)
    M[3, 2] = -1
    return M


def main():
    # Main function to initialize the GLFW window, set up OpenGL context, and enter the render loop.
    # Initialize GLFW and configure OpenGL context.
    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create a window for rendering.
    SCREEN_WIDTH = 720
    SCREEN_HEIGHT = 480
    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "OpenGL Window", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Enable depth testing and set viewport size.
    glEnable(GL_DEPTH_TEST)
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

    # Enable blending for handling transparency.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Load and prepare textured meshes for rendering.
    mesh_files = [("Bottles.ply", "bottles.bmp"), ("Floor.ply", "floor.bmp"), ("Patio.ply", "patio.bmp"),
                  ("Table.ply", "table.bmp"), ("Walls.ply", "walls.bmp"), ("WindowBG.ply", "windowbg.bmp"),
                  ("WoodObjects.ply", "woodobjects.bmp"), ("DoorBG.ply", "doorbg.bmp"),
                  ("MetalObjects.ply", "metalobjects.bmp"), ("Curtains.ply", "curtains.bmp")]
    meshes = [TexturedMesh("RenderSceneAssets/" + plyFilePath, "RenderSceneAssets/" + bmpFilePath) for
              plyFilePath, bmpFilePath in mesh_files]

    while not glfw.window_should_close(window):
        # Render loop: process input, clear screen, and draw the scene.
        # Per-frame time logic
        global delta_time, last_frame
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame

        # Input
        process_input(window)

        glClearColor(0.1, 0.1, 0.1, 1.0)  # A non-black color for clarity, e.g., dark gray
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view = lookAt(camera_pos, camera_pos + camera_front, camera_up)
        projection = perspective(np.radians(45), SCREEN_WIDTH / SCREEN_HEIGHT, 0.1, 100)

        for mesh in meshes:
            MVP = (projection @ view @ np.identity(4)).transpose()  # Transpose so MVP matrix is column major
            mesh.draw(MVP)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
