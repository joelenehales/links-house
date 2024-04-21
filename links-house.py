from OpenGL.GL import *
import glfw
import glm
import os
import struct
import numpy as np


# Initialize glfw
glfw.init() 
glfw.window_hint(glfw.SAMPLES, 4)  # Enable 4x multisampling

# Set context version required for shaders and VAOs
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # For MacOS
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

# Open a window and create its OpenGL context
screen_width = 1400
screen_height = 900

window = glfw.create_window(screen_width, screen_height, "Links House", None, None)
glfw.make_context_current(window)


# File opening helper functions
def readPLYFile(filename):
	""" Reads a PLY file.
	
	Parameters
	----------
	filename : str
		Filepath to a PLY file.
	
	Returns
	-------
	vertex_objects, triangle_objects : list[VertexData], list[TriData]
		Lists of vertices and triangle faces in the PLY file
	
	"""

	# Get filepath to file
	directory = os.path.dirname(os.path.abspath(__file__))
	filepath = os.path.join(directory, filename)

	with open(filepath, "r") as file:

		# Read formatting and comments
		filetype = file.readline().strip("\n")
		format = file.readline().strip("\n")
		comment = file.readline().strip("\n")

		# Read number of vertices
		line = file.readline().strip().split(" ")
		num_vertices = int(line[2])

		# Create empty arrays to store each vertex's attributes
		vertex_data = {"x" : np.empty(num_vertices),
		               "y" : np.empty(num_vertices),
		               "z" : np.empty(num_vertices),
		               "nx" : np.empty(num_vertices),
		               "ny" : np.empty(num_vertices),
		               "nz" : np.empty(num_vertices),
		               "r" : np.empty(num_vertices),
		               "g" : np.empty(num_vertices),
		               "b" : np.empty(num_vertices),
		               "u" : np.empty(num_vertices),
		               "v" : np.empty(num_vertices)}

		# Determine which value in each vertex data line corresponds to each attribute
		attribute_indices = {}  # Stores each index in the line and the attribute it corresponds to
		line_num = 0            # Used to iterate until the end of the vertex property information
		while (line_num < len(vertex_data)): 
			
			line = file.readline().strip().split()  # Split the line's data into components

			if (line[0] == "element"):  # Beginning of triangle face properties begin
				break
			else:
				# Store the current position and its corresponding attribute
				attribute_indices[line_num] = line[2]
				line_num += 1

		# Read number of faces
		num_faces = int(line[2])

		# Skip format of triangle faces and end of header
		next(file)
		next(file)

		# Read vertex data from file
		for index in range(num_vertices):  # Iterate over each vertex

			vertex = file.readline().strip().split(" ")  # Split the line's data into components

			for i in range(len(vertex)):  # Iterate over each attribute

				attribute = attribute_indices[i]  # Lookup which attribute the value in this position corrresponds to
				vertex_data[attribute][index] = float(vertex[i])  # Add attribute to correct list


		# Create vertex objects from read vertex data
		vertex_objects = []
		for i in range(num_vertices):
			
			position = (vertex_data['x'][i], 
			            vertex_data['y'][i], 
			            vertex_data['z'][i])

			normal = (vertex_data['nx'][i], 
			          vertex_data['ny'][i], 
			          vertex_data['nz'][i])

			color = (vertex_data['r'][i],
			         vertex_data['g'][i],
			         vertex_data['b'][i])
			
			texture = (vertex_data['u'][i],
			           vertex_data['v'][i])

			vertex = VertexData(position, normal, color, texture)
			vertex_objects.append(vertex)


		# Read triangle face indices from file and create triangle objects
		triangle_objects = []
		while(len(triangle_objects) < num_faces):  # Repeat until all faces have been read
			
			# Read indices from file
			face = file.readline().strip().split(" ")  # Split line into 3 indices
			indices = [int(index) for index in face[1:]]  # Convert each index to an integer and add to list

			# Create triangle object
			triangle = TriData(indices)
			triangle_objects.append(triangle)

		return vertex_objects, triangle_objects


def readBitmapFile(filename):
	""" Reads a bitmap image file.
	
	Parameters
	----------
	filename : str
		Filepath to a bitmap image file.
	
	Returns
	-------
	bitmap_image : array[float]
		Bitmap image
	width : int
		Width of image, in bytes
	height : int
		Height of image, in bytes

	"""

	# Get filepath to file
	directory = os.path.dirname(os.path.abspath(__file__))
	filepath = os.path.join(directory, filename)

	with open(filepath, 'rb') as bmp:  # Open file

		# Unpack all header data
		type = bmp.read(2).decode()                              # Type of image file
		image_size = struct.unpack('I', bmp.read(4))[0]          # Filesize in bytes
		struct.unpack('xxxxxxxx', bmp.read(8))                   # Skip 8 bytes
		header_size = struct.unpack('I', bmp.read(4))[0]         # Header size, in bytes
		width = struct.unpack('I', bmp.read(4))[0]               # Image width, in bytes
		height = struct.unpack('I', bmp.read(4))[0]              # Image height, in bytes
		struct.unpack('xx', bmp.read(2))                         # Skip 2 bytes
		bits_per_pixel = struct.unpack('H', bmp.read(2))[0]      # Bits per pixel
		struct.unpack('xxxxxxxxxxxxxxxxxxxxxxxx', bmp.read(24))  # Skip 24 bytes

		# Error handling
		if (type != "BM"):  # Bitmap image files always begin with "BM"
			raise ValueError("Incorrect file format. Must be a .bmp file.")

		if (bits_per_pixel != 32):  # Ensure file is 32 bits per pixel
			raise ValueError("Incorrect bitmap file. Must be 32 bits per pixel.")

		if (image_size == 0):  # Misformatted image size
			image_size = width * height * 4   # x4 for each channel (R, G, B, A)

		# Read image data
		image_data = []
		for byte in range(image_size):
			image_data.append(int.from_bytes(bmp.read(1)))  # Read one byte

	return image_data, width, height


# Data structures to store vertex data, triangle faces, textured triangle meshes
class VertexData():
	""" Class representing a vertex.
	
	Attributes
	----------
	position : tuple(float, float, float)
		Vertex position (x, y, z)
	normal : tuple(float, float, float)
		Normal vector (x, y, z)
	color : tuple(float, float, float)
		Color (RGB)
	texture : tuple(float, float)
		Texture coordinates (u, v)

	"""

	

	def __init__(self, position, normal=None, color=None, texture=None):
		""" Creates a new vertex with the given position and optional vertex
		attribues. 
		
		Parameters
		----------
		position : tuple(float, float, float)
			Vertex position (x, y, z)
		normal : tuple(float, float, float)
			Normal vector (x, y, z)
		color : tuple(float, float, float)
			Color (RGB)
		texture : tuple(float, float)
			Texture coordinates (u, v)

		"""

		# Initialize attributes
		x, y, z = position
		self.position = glm.vec3(x, y, z)   # Vertex position

		nx, ny, nz = normal
		self.normal = glm.vec3(nx, ny, nz)  # Normal vector

		r, g, b = color
		self.color = (r, g, b)              # Color

		u, v = texture
		self.texture = glm.vec2(u, v)       # Texture coordinates



class TriData():
	""" Class representing a triangle.
	
	Attributes
	----------
	indices : list[int, int, int]
	    Indices of the three vertices that make up the triangle. Must have a
	    length of 3.

	"""

	def __init__(self, indices):
		""" Creates a new triangle with the given vertex indices.
		
		Parameters
		----------
		indices : list[int, int, int]
			Indices of the three vertices that make up the triangle. Must have a
			length of 3.

		"""

		# Initialize attributes
		self.indices = indices


class TexturedMesh():
	""" Class representing a textured triangle mesh. 
	
	Attributes
	----------
	vertex_VBO : int
	    Integer ID of the vertex buffer object storing the positions of each
	    vertex in the triangle mesh.
	texture_VBO : int
	    Integer ID of the vertex buffer object storing the texture coordinates
	    of each vertex in the triangle mesh.
	face_EBO : int
	    Integer ID of the element buffer object storing the indices that make up
	    each face in the triangle mesh.
	texture_ID : int
		Integer ID of the texture created from the bitmap image.
	VAO : int
		Integer ID of the vertex array object used to render the triangle mesh.
	program_ID : int
	    Integer ID of the shader program created and linked to render the
	    triangle mesh.
	MVP : int
	    Integer ID for model view projection matrix uniform variable in shader
	    program.
		
	Methods
	-------
	draw(self, MVP):
		Renders the texture mesh object.

	"""

	def __init__(self, PLY_file, bitmap_file):
		""" Creates a textured triangle mesh from a given PLY file and bitmap
		image file.
		
		Parameters
		----------
		PLY_file : str
			Filepath to a PLY file.
		bitmap_file : str
			Filepath to a bitmap image file.

		"""

		# Read vertex and face data from PLY file
		vertex_objects, face_objects = readPLYFile(PLY_file)

		# Extract vertex positions and texture coordinates
		vertex_data = []
		texture_coordinates = []
		for vertex in vertex_objects:

			vertex_data.append(vertex.position.x)  # Vertex position
			vertex_data.append(vertex.position.y)
			vertex_data.append(vertex.position.z)

			texture_coordinates.append(vertex.texture.x)  # Texture coordinates
			texture_coordinates.append(vertex.texture.y)

		# Extract vertex face indices
		face_indices = []
		for face in face_objects:
			face_indices.append(face.indices[0])
			face_indices.append(face.indices[1])
			face_indices.append(face.indices[2])

		# Convert to arrays
		vertex_data = np.array(vertex_data, dtype=np.float32)
		texture_coordinates = np.array(texture_coordinates, dtype=np.float32)
		face_indices = np.array(face_indices, dtype=np.uint32)

		# Load bitmap image
		bitmap_image, texture_width, texture_height = readBitmapFile(bitmap_file)

		# Define shader codes (From L10texture.cpp)
		vertex_shader_code = """\
    	#version 330 core\n\
		// Input vertex data, different for all executions of this shader.\n\
		layout(location = 0) in vec3 vertexPosition;\n\
		layout(location = 1) in vec2 uv;\n\
		// Output data ; will be interpolated for each fragment.\n\
		out vec2 uv_out;\n\
		// Values that stay constant for the whole mesh.\n\
		uniform mat4 MVP;\n\
		void main(){ \n\
			// Output position of the vertex, in clip space : MVP * position\n\
			gl_Position =  MVP * vec4(vertexPosition,1);\n\
			// The color will be interpolated to produce the color of each fragment\n\
			uv_out = uv;\n\
		}\n"""

		fragment_shader_code = """\
		#version 330 core\n\
		in vec2 uv_out; \n\
		uniform sampler2D tex;\n\
		out vec4 fragColor;\n\
		void main() {\n\
			fragColor = texture(tex, uv_out);\n\
		}\n"""

		# Create vertex and fragment shaders
		vertex_shader_ID = glCreateShader(GL_VERTEX_SHADER)
		fragment_shader_ID = glCreateShader(GL_FRAGMENT_SHADER)

		# Compile Vertex Shader
		glShaderSource(vertex_shader_ID, vertex_shader_code)
		glCompileShader(vertex_shader_ID)

		# Check for compilation error
		if not(glGetShaderiv(vertex_shader_ID, GL_COMPILE_STATUS)):
			raise RuntimeError(glGetShaderInfoLog(vertex_shader_ID))

		# Compile Fragment Shader
		glShaderSource(fragment_shader_ID, fragment_shader_code)
		glCompileShader(fragment_shader_ID)

		# Check for compilation error
		if not(glGetShaderiv(fragment_shader_ID, GL_COMPILE_STATUS)):
			raise RuntimeError(glGetShaderInfoLog(fragment_shader_ID))

		# Link shader program and attach shaders
		self.program_ID = glCreateProgram()

		glAttachShader(self.program_ID, vertex_shader_ID)
		glAttachShader(self.program_ID, fragment_shader_ID)
		glLinkProgram(self.program_ID)

		glDetachShader(self.program_ID, vertex_shader_ID)
		glDetachShader(self.program_ID, fragment_shader_ID)

		glDeleteShader(vertex_shader_ID)
		glDeleteShader(fragment_shader_ID)

		# Get handle for model view projection matrix uniform
		self.MVP = glGetUniformLocation(self.program_ID, "MVP")

		# Create texture from bitmap image
		self.texture_ID = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.texture_ID)

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture_width, texture_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, bitmap_image)
		glGenerateMipmap(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, 0)

		# Create and bind VAO
		self.VAO = glGenVertexArrays(1)
		glBindVertexArray(self.VAO)

		# Create and bind VBO for vertex data
		self.vertex_VBO = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.vertex_VBO)
		glBufferData(GL_ARRAY_BUFFER, np.dtype(np.float32).itemsize*len(vertex_data), vertex_data, GL_STATIC_DRAW)

		# Set vertex attributes
		glEnableVertexAttribArray(0)
		glVertexAttribPointer(
			0,            # Attribute number
			3,            # Size (Number of components)
			GL_FLOAT,     # Type
			GL_FALSE,     # Normalized?
			0,            # Stride (Byte offset)
			None          # Offset
		)

		# Create and bind VBO for texture data
		self.texture_VBO = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.texture_VBO)
		glBufferData(GL_ARRAY_BUFFER, np.dtype(np.float32).itemsize*len(texture_coordinates), texture_coordinates, GL_STATIC_DRAW)

		# Set texture attributes
		glEnableVertexAttribArray(1)
		glVertexAttribPointer(
			1,            # Attribute number
			2,            # Size (Number of components)
			GL_FLOAT,     # Type
			GL_FALSE,     # Normalized?
			0,            # Stride (Byte offset)
			None          # Offset
		)

		# Create and bind EBO for face indices
		self.face_EBO = glGenBuffers(1)
		self.num_indices = len(face_indices)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.face_EBO)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.dtype(np.int32).itemsize*self.num_indices, face_indices, GL_STATIC_DRAW)

		glBindVertexArray(0)  # Unbind VAO


	def draw(self, MVP):
		""" Renders the texture mesh object.
		
		Parameters
		----------
		MVP : glm.mat4
			Model view projection matrix.

		"""

		# Enable blending
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		# Bind texture
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture_ID)

		# Use program and set uniforms
		glUseProgram(self.program_ID)
		MVP_list = np.array([MVP[i][j] for i in range(4) for j in range(4)])
		glUniformMatrix4fv(self.MVP, 1, GL_FALSE, MVP_list)

		# Bind VAO to restore captured state (buffer bindings and attribute specifications)
		glBindVertexArray(self.VAO)

		# Draw triangles
		glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)

		# Unbind VAO and texture, clean up shader program
		glBindVertexArray(0)
		glBindTexture(GL_TEXTURE_2D, 0)
		glUseProgram(0)

		glDisable(GL_BLEND)


# Create all textured triangle meshes. List semi-transparent objects at the end
# so they are rendered last.
PLY_files = ["WindowBG.ply", "Patio.ply", "Walls.ply", "Floor.ply", "Table.ply", "WoodObjects.ply", "Bottles.ply", "MetalObjects.ply", "DoorBG.ply", "Curtains.ply"]
bitmap_files = ["windowbg.bmp", "patio.bmp", "walls.bmp", "floor.bmp", "table.bmp", "woodobjects.bmp", "bottles.bmp", "metalobjects.bmp", "doorbg.bmp", "curtains.bmp"]

object_meshes = []

for PLY_file, bitmap_file in zip(PLY_files, bitmap_files):
	object_mesh = TexturedMesh("Assets/"+PLY_file, "Assets/"+bitmap_file)
	object_meshes.append(object_mesh)


# Define matrices
P = glm.perspective(glm.radians(45.0), screen_width/screen_height, 0.001, 1000.0)  # Projection matrix with a vertical field of view of 45 degrees
M = glm.mat4(1.0)  # Model matrix

# Initialize camera settings
eye = glm.vec3(0.0, 0.0, -1.0)     # Camera look direction
up = glm.vec3(0.0, 1.0, 0.0)       # Direction of up
center = glm.vec3(0.5, 0.4, 0.5)   # Camera position

# Set values used to move camera each frame
translate_units = 0.02   # Units translated each frame up/down arrow key is held
rotate_angle = 1         # Angle rotated each frame left/right arrow key is held, in degrees

# Ensure depth is determined correctly
glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)

# Set background to light green
glClearColor(0.83, 0.85, 0.63, 0.0)

# Ensure we can capture the escape key being pressed
glfw.set_input_mode(window, glfw.STICKY_KEYS, GL_TRUE)

# Render loop. Repeat until escape key is pressed or window is closed
while (glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window)):

	# Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	glfw.poll_events()  # Poll for arrow key presses and move camera accordingly

	# Check if up or down arrows are pressed. Only allow one to be pressed at a time.
	if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):  # Up arrow pressed
		center = center + translate_units * eye  # Move camera forward in the direction currently facing

	elif (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):  # Down arrow pressed
		center = center - translate_units * eye  # Move camera backward in the direction currently facing

	else:
		pass

	# Check if left or right arrows are pressed. Only allow one to be pressed at a time.
	if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):  # Left arrow pressed

		rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(rotate_angle), up)  # Define rotation matrix around the direction of up
		eye = glm.vec3(rotation_matrix * glm.vec4(eye, 1.0))  # Apply rotation to look direction

	elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):  # Right arrow pressed

		rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-rotate_angle), up)  # Define rotation matrix around the direction of up
		eye = glm.vec3(rotation_matrix * glm.vec4(eye, 1.0))  # Apply rotation to look direction

	else:
		pass

	# Set view matrix using new camera position
	V = glm.lookAt(center, eye + center, up); 

	# Calculate model view projection matrix
	MVP = P * V * M

	# Render each textured triangle mesh object
	for object_mesh in object_meshes:
		object_mesh.draw(MVP)

	# Swap buffers
	glfw.swap_buffers(window)
	glfw.poll_events()

# Close OpenGL window and terminate GLFW
glfw.terminate()
