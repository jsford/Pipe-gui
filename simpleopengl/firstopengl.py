#!/usr/bin/python
SCREEN_SIZE = (1200, 740)

import numpy as np
from math import radians 
import colorsys

from OpenGL.GL import *
from OpenGL.GLU import *

import pygame
from pygame.locals import *

from gameobjects.matrix44 import *
from gameobjects.vector3 import *

from terrain.generator import *

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
    
    glShadeModel( GL_SMOOTH )
    glClearColor(0.0, 0.0, 0.0, 0.0)

    glEnable(GL_COLOR_MATERIAL)
    
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)        
    glLight(GL_LIGHT0, GL_POSITION,  (0, 1, 1, 0))    


def interp_color(c1, c2, t):
    h1, s1, v1 = colorsys.rgb_to_hsv(c1[0], c1[1], c1[2])
    h2, s2, v2 = colorsys.rgb_to_hsv(c2[0], c2[1], c2[2])
    h3 = t*h1+(1-t)*h2
    s3 = t*s1+(1-t)*s2
    v3 = t*v1+(1-t)*v2
    return colorsys.hsv_to_rgb(h3, s3, v3)


class Pipe(object):
    def __init__(self, length, rad, pipe_color, deposit_color):
        self.rad = rad
        self.pipe_color = pipe_color
        self.deposit_color = deposit_color
        self.length = length
        self.axis = Vector3.from_floats(0, 0, -1)
        self.display_list_pipe = None
        self.display_list_deposit = None

        # Create the deposit surface using the
        # diamond-square terrain generation algorithm.
        self.deposits = diamond_square_alg(7)
        self.threshold = 0.5
        self.deposits = threshold_map(self.deposits, self.threshold)
        self.deposits *= in2m(1.0) 

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


            glBegin(GL_TRIANGLES)
            
            max_dep = np.amax(self.deposits)
            for s in range(0, 100-1):
                for t in range(0, 100):
                    theta = t*2*np.pi/100.0
                    glNormal3dv( self.deposit_norms[s*100+t] )

                    t_plus_1 = (t+1)%100


                    if ((self.deposits[s, t]       > 0.00001) or
                       (self.deposits[s+1, t]      > 0.00001) or
                       (self.deposits[s, t_plus_1] > 0.00001)):

                        c1 = interp_color(self.deposit_color, self.pipe_color, self.deposits[s, t]/max_dep)
                        c2 = interp_color(self.deposit_color, self.pipe_color, self.deposits[s+1, t]/max_dep)
                        c3 = interp_color(self.deposit_color, self.pipe_color, self.deposits[s, t_plus_1]/max_dep)

                        glColor( c1 )
                        glVertex( self.deposit_verts[    s * 100 + t  ] )
                        glColor( c2 )
                        glVertex( self.deposit_verts[(s+1) * 100 + t  ] )
                        glColor( c3 )
                        glVertex( self.deposit_verts[    s * 100 + t_plus_1 ] )

                    if ((self.deposits[s+1, t_plus_1] > 0.00001) or
                       (self.deposits[s+1, t]         > 0.00001) or
                       (self.deposits[s, t_plus_1]    > 0.00001)):

                        c1 = interp_color(self.deposit_color, self.pipe_color, self.deposits[s+1, t_plus_1]/max_dep)
                        c2 = interp_color(self.deposit_color, self.pipe_color, self.deposits[s+1, t]/max_dep)
                        c3 = interp_color(self.deposit_color, self.pipe_color, self.deposits[s, t_plus_1]/max_dep)

                        glColor( c1 )
                        glVertex( self.deposit_verts[(s+1) * 100 + t_plus_1 ] )
                        glColor( c2 )
                        glVertex( self.deposit_verts[(s+1) * 100 + t ] )
                        glColor( c3 )
                        glVertex( self.deposit_verts[    s * 100 + t_plus_1 ] )

            glEnd()
    
            glEndList()

        else:
            
            # Render the display list            
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
    pipe = Pipe(10, in2m(30/2.0), (0, 0, 1), (1,0,0))

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
        movement_speed = ft2m(10)/60.0
        
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
            movement_speed = 1.0
        elif pressed[K_a]:
            movement_direction.z = +5.0
            movement_speed = 1.0
        
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
