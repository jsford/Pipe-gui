#!/usr/bin/python
SCREEN_SIZE = (1600, 1200)

import numpy as np
from math import radians 

from OpenGL.GL import *
from OpenGL.GLU import *

import pygame
from pygame.locals import *

from gameobjects.matrix44 import *
from gameobjects.vector3 import *

def ft2m(feet):
    return feet*0.3048
def in2m(inches):
    return inches*0.0254

def resize(width, height):
    
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, float(width)/height, .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    
    glEnable(GL_DEPTH_TEST)
    
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
    glClearColor(0.0, 0.0, 0.0, 0.0)

    glEnable(GL_COLOR_MATERIAL)
    
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)        
    glLight(GL_LIGHT0, GL_POSITION,  (0, 1, 1, 0))    

class Pipe(object):
    def __init__(self, length, rad, pipe_color, deposit_color):
        self.rad = rad
        self.pipe_color = pipe_color
        self.deposit_color = deposit_color
        self.length = length
        self.axis = Vector3.from_floats(0, 0, -1)
        self.display_list_pipe = None
        self.display_list_deposit = None

        self.deposits = in2m(1)*np.random.rand(100,100)
        self.deposits[self.deposits < 0] = 0

        self.pipe_verts = []*100
        self.pipe_norms = []*100

        self.deposit_verts = []*100
        self.deposit_norms = []*100

        for s in range(0, 100):
            center = tuple(self.axis*(s/100.0*self.length) + Vector3(0,0,0))
            for t in range(0, 100):
                theta = t*2*np.pi/100.0
                self.pipe_verts.append( (center[0]+self.rad*np.cos(theta), center[1]+self.rad*np.sin(theta), center[2]) )
                self.pipe_norms.append( tuple(Vector3.from_points(self.pipe_verts[-1], center)) ) 
                r = self.rad - self.deposits[s,t]
                self.deposit_verts.append( (center[0]+r*np.cos(theta), center[1]+r*np.sin(theta), center[2]) )

        self.deposit_norms = self.pipe_norms

        

    def render(self):

        if self.display_list_deposit is None:
            
            
            # Create a display list
            self.display_list_deposit = glGenLists(1)                
            glNewList(self.display_list_deposit, GL_COMPILE)

            glColor( self.deposit_color )

            glBegin(GL_TRIANGLES)
            
            for s in range(0, 100-1):
                for t in range(0, 100):
                    theta = t*2*np.pi/100.0
                    glNormal3dv( self.deposit_norms[s*100+t] )

                    t_plus_1 = (t+1)%100
                    glVertex( self.deposit_verts[    s * 100 + t  ] )
                    glVertex( self.deposit_verts[(s+1) * 100 + t  ] )
                    glVertex( self.deposit_verts[    s * 100 + t_plus_1 ] )

                    glVertex( self.deposit_verts[(s+1) * 100 + t_plus_1 ] )
                    glVertex( self.deposit_verts[(s+1) * 100 + t ] )
                    glVertex( self.deposit_verts[    s * 100 + t_plus_1 ] )

            glEnd()
    
            glEndList()

        else:
            
            # Render the display list            
            glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
            glCallList(self.display_list_deposit)

        if self.display_list_pipe is None:
            
            # Create a display list
            self.display_list_pipe = glGenLists(1)                
            glNewList(self.display_list_pipe, GL_COMPILE)

            glColor( self.pipe_color )

            glBegin(GL_TRIANGLES)
            
            for s in range(0, 100-1):
                for t in range(0, 100):
                    theta = t*2*np.pi/100.0
                    glNormal3dv( self.pipe_norms[s*100+t] )

                    t_plus_1 = (t+1)%100
                    glVertex( self.pipe_verts[    s * 100 + t  ] )
                    glVertex( self.pipe_verts[(s+1) * 100 + t  ] )
                    glVertex( self.pipe_verts[    s * 100 + t_plus_1 ] )

                    glVertex( self.pipe_verts[(s+1) * 100 + t_plus_1 ] )
                    glVertex( self.pipe_verts[(s+1) * 100 + t ] )
                    glVertex( self.pipe_verts[    s * 100 + t_plus_1 ] )

            glEnd()
    
            glEndList()

        else:
            
            # Render the display list            
            glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )
            glCallList(self.display_list_pipe)

def run():
    
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE, HWSURFACE|OPENGL|DOUBLEBUF)
    
    resize(*SCREEN_SIZE)
    init()
    
    clock = pygame.time.Clock()    
    
    glMaterial(GL_FRONT, GL_AMBIENT, (0.1, 0.1, 0.1, 1.0))    
    glMaterial(GL_FRONT, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))

    # This object renders the pipe
    pipe = Pipe(10, in2m(30/2.0), (1, 0, 0), (0,0,1))

    # Camera transform matrix
    camera_matrix = Matrix44()
    camera_matrix.translate = (0.0, 0.0, 0.8)

    # Initialize speeds and directions
    rotation_direction = Vector3()
    rotation_speed = radians(90.0)
    movement_direction = Vector3()
    movement_speed = ft2m(10)/60.0

    while True:
        
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            if event.type == KEYUP and event.key == K_ESCAPE:
                return                
            
        # Clear the screen, and z-buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                        
        time_passed = clock.tick()
        time_passed_seconds = time_passed / 1000.
        
        pressed = pygame.key.get_pressed()
        
        # Reset rotation and movement directions
        rotation_direction.set(0.0, 0.0, 0.0)
        movement_direction.set(0.0, 0.0, -1.0)
        
        # Modify direction vectors for key presses
        if pressed[K_LEFT]:
            rotation_direction.y = +1.0
        elif pressed[K_RIGHT]:
            rotation_direction.y = -1.0
        if pressed[K_UP]:
            rotation_direction.x = -1.0
        elif pressed[K_DOWN]:
            rotation_direction.x = +1.0
        if pressed[K_z]:
            rotation_direction.z = -1.0
        elif pressed[K_x]:
            rotation_direction.z = +1.0            
        if pressed[K_q]:
            movement_direction.z = -5.0
        elif pressed[K_a]:
            movement_direction.z = +5.0
        
        # Calculate rotation matrix and multiply by camera matrix    
        rotation = rotation_direction * rotation_speed * time_passed_seconds
        rotation_matrix = Matrix44.xyz_rotation(*rotation)        
        camera_matrix *= rotation_matrix
        
        # Calcluate movment and add it to camera matrix translate
        heading = Vector3(camera_matrix.forward)
        movement = heading * movement_direction.z * movement_speed                    
        camera_matrix.translate += movement * time_passed_seconds
        
        # Upload the inverse camera matrix to OpenGL
        glLoadMatrixd(camera_matrix.get_inverse().to_opengl())
                
        # Light must be transformed as well
        glLight(GL_LIGHT0, GL_POSITION,  (0, 1.5, 1, 0)) 
                
        # Render the pipe
        pipe.render()
        
        # Show the screen
        pygame.display.flip()

run()
