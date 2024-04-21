============
Link's House
============

Python OpenGL program that renders the inside of Link's House from *The Legend of
Zelda: Ocarina of Time*. The rendered world can be explored by pressing the up,
down, left, and right arrow keys to move forwards, backwards, left, and right
respectively.


Preview
-------

.. image:: program-screenshot-1.png
  :width: 675
  :alt: Program Screenshot 1

.. image:: program-screenshot-2.png
  :width: 675
  :alt: Program Screenshot 2



Requirements
------------

The program requires ``pyopengl``, ``glfw``, ``pyglm``, and ``numpy``. For help
setting up a new Python environment and downloading the packages, check out my
beginner PyOpenGL tutorial, `Hello Triangle<https://github.com/joelenehales/hello-triangle>`_!


Compiling the Program
---------------------

Download and unzip the project. Once the required packages have been installed
and the correct environment has been activated, the program can be run by
running ``links-house.py`` from any Python IDE, or from the command line using:

.. code-block:: bash

    python links-house.py


Explaination of Source Code
---------------------------

Creating Triangle Meshes
^^^^^^^^^^^^^^^^^^^^^^^^

The program leverages the object-oriented nature of Python to structure the
code's data, representing vertex data, triangle faces, and textured triangle
meshes using classes. The program begins by creating a textured triangle mesh
object for each item in the scene. The class constructor loads the vertex data
and triangle face indices, compiles the shader program, and generates the item's
texture from a bitmap image. To simplify the specification of vertex attributes,
a vertex array object (VAO) is initialized once in the constructor. With the VAO
bound, the constructor binds vertex buffer objects (VBOs) that store the
position and texture coordinates of the triangle mesh's vertices, as well as an
element buffer object (EBO) that stores the vertices making up each face in the
tringle mesh. Each buffer object uses GL_STATIC_DRAW, allowing it to be drawn
multiple times. To avoid accidentally capturing an unintended state, the VAO is
unbound at the end of the constructor once all vertex attributes have been
specified.


Camera Movement
^^^^^^^^^^^^^^^

Before the render loop, the program defines the projection matrix and initializes
the camera settings. Within the render loop, the program first polls for arrow
key presses. For each frame the up or down arrow is pressed, the program moves
0.02 units forward or backward respectively. For each frame the left or right
arrow is pressed, the program rotates 1 degree counter-clockwise or clockwise
respectively. These values were chosen to ensure the camera movement feels
smooth and natural for the world size. The program allows the user to move
forward or backward and left or right at the same time, to simulate real-life movement.


Rendering the Scene
^^^^^^^^^^^^^^^^^^^

After moving the camera and setting the model view projection matrix, the
program then renders each item in the scene using its previously created
textured triangle mesh. On each frame, the program binds the VAO created in the
constructor to restore the captured state, including all buffer bindings and
vertex attribute specifications. The item is rendered by draiwng the vertices in
order as indicated by the EBO. After the object has been rendered, the program unbinds
the VAO and texture, and cleans up the shader program.


Known Issues
------------

Texture Mapping
^^^^^^^^^^^^^^^

Minor discrepancies can be seen in the way some bitmap images map to their
corresponding triangle meshes. This is likely due to the format of the bitmap
images being unable to be read by standard Python image packages, requiring
custom functions to be written.
