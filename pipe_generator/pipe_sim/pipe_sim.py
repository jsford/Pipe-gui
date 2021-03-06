#!/usr/bin/python
SCREEN_SIZE = (800, 600)

import numpy as np
from math import radians 
import colorsys
import sys

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
    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)        
    glLight(GL_LIGHT0, GL_POSITION,  (0, 1, 1, 0))    


def interp_color(c1, c2, t):


    # Interpolate the color based on deposit depth

    h1, s1, v1 = colorsys.rgb_to_hsv(c1[0], c1[1], c1[2])
    h2, s2, v2 = colorsys.rgb_to_hsv(c2[0], c2[1], c2[2])
    h3 = t*h1+(1-t)*h2
    s3 = t*s1+(1-t)*s2
    v3 = t*v1+(1-t)*v2
    r, g, b = colorsys.hsv_to_rgb(h3, s3, v3)
    alpha = 1.0

    # Blend the edges using a shape like this
    #
    # ---------
    #          \
    #           \
    #            \
    #             \___
    #
    # Interpolate alpha down to zero.
    # Interpolate color down to the pipe color.

    if t < 0.1:
        alpha = t*c1[3] + (1-t)*c2[3]
        t /= 0.1
        h1, s1, v1 = colorsys.rgb_to_hsv(r, g, b)
        h2, s2, v2 = colorsys.rgb_to_hsv(231/255., 231/255., 231/255.)
        h3 = t*h1+(1-t)*h2
        s3 = t*s1+(1-t)*s2
        v3 = t*v1+(1-t)*v2
        r, g, b = colorsys.hsv_to_rgb(h3, s3, v3)

    return (r, g, b, alpha)


class Pipe(object):
    def __init__(self, length, rad, deposit_seed=None):
        self.rad = rad
        self.pipe_color = (231/255., 231/255., 231/255.)
        self.length = length
        self.axis = Vector3.from_floats(0, 0, -1)
        self.display_list_pipe = None
        self.display_list_deposit = None

        # Create the deposit surface using the
        # diamond-square terrain generation algorithm.
        
        self.deposits = DepositGenerator(128, 1024, threshold=0.8).pipe_map.transpose()
        self.deposits *= in2m(1.0) 

        self.pipe_verts = np.zeros((1024, 128, 3))
        self.pipe_norms = np.zeros((1024, 128, 3))

        self.deposit_verts = np.zeros((1024, 128, 3))
        self.deposit_norms = np.zeros((1024, 128, 3))

        for s in range(0, 1024):
            center = tuple(self.axis*(s/1024.0*self.length) + Vector3(0,0,0))
            for t in range(0, 128):
                theta = t*2*np.pi/128.0
                self.pipe_verts[s,t,:] = (center[0]+self.rad*np.cos(theta), center[1]+self.rad*np.sin(theta), center[2])
                self.pipe_norms[s,t,:] = (tuple(Vector3.from_points(self.pipe_verts[s,t,:], center))) 
                r = self.rad - self.deposits[s,t]
                self.deposit_verts[s,t] = (center[0]+r*np.cos(theta), center[1]+r*np.sin(theta), center[2])

        self.deposit_norms = self.pipe_norms

        

    def render(self):

        if self.display_list_pipe is None:
            
            # Create a display list
            self.display_list_pipe = glGenLists(1)                
            glNewList(self.display_list_pipe, GL_COMPILE)

            glBegin(GL_TRIANGLES)
            
            for s in range(0, 1024-1):
                for t in range(0, 128):
                    t_plus_1 = (t+1)%128

                    
                    #if s%16 == 0:
                    #    glColor((0,0,1))
                    #else:
                    #    glColor(self.pipe_color)
                    glColor(self.pipe_color)

                    glNormal3dv( self.pipe_norms[s, t, :] )
                    glVertex( self.pipe_verts[s, t, :] )

                    glNormal3dv( self.pipe_norms[s+1, t, :] )
                    glVertex( self.pipe_verts[s+1, t, :] )

                    glNormal3dv( self.pipe_norms[s, t_plus_1, :] )
                    glVertex( self.pipe_verts[s, t_plus_1, :] )

                    glNormal3dv( self.pipe_norms[s+1, t, :] )
                    glVertex( self.pipe_verts[s+1, t, :] )

                    glNormal3dv( self.pipe_norms[s+1, t_plus_1, :] )
                    glVertex( self.pipe_verts[s+1, t_plus_1, :] )

                    glNormal3dv( self.pipe_norms[s, t_plus_1, :] )
                    glVertex( self.pipe_verts[s, t_plus_1, :] )

            glEnd()
    
            glEndList()

        else:
            
            # Render the display list            
            glCallList(self.display_list_pipe)


        if self.display_list_deposit is None:
            
            # Create a display list
            self.display_list_deposit = glGenLists(1)                
            glNewList(self.display_list_deposit, GL_COMPILE)


            glBegin(GL_TRIANGLES)
            
            max_dep = np.amax(self.deposits)
            min_dep = 0
            rng_dep = (max_dep-min_dep)

            for s in range(0, 1024-1):
                for t in range(0, 128):

                    t_plus_1 = (t+1)%128


                    if ((self.deposits[  s, t]      > 0.0001) or
                        (self.deposits[s+1, t]      > 0.0001) or
                        (self.deposits[s, t_plus_1] > 0.0001)):

                        c1 = interp_color((1, 0, 0, 1), (0, 1, 0, 0), (self.deposits[s, t]-min_dep)/rng_dep)
                        c2 = interp_color((1, 0, 0, 1), (0, 1, 0, 0), (self.deposits[s+1, t]-min_dep)/rng_dep)
                        c3 = interp_color((1, 0, 0, 1), (0, 1, 0, 0), (self.deposits[s, t_plus_1]-min_dep)/rng_dep)


                        glColor4f( *c1 )
                        glNormal3dv( self.deposit_norms[s, t, :] )
                        glVertex( self.deposit_verts[s, t, :] )
                        glColor4f( *c2 )
                        glNormal3dv( self.deposit_norms[s+1, t, :] )
                        glVertex( self.deposit_verts[s+1, t, :] )
                        glColor4f( *c3 )
                        glNormal3dv( self.deposit_norms[s, t_plus_1, :] )
                        glVertex( self.deposit_verts[s, t_plus_1, :] )

                    if ((self.deposits[s+1, t_plus_1] > 0.0001) or
                        (self.deposits[s+1, t]        > 0.0001) or
                        (self.deposits[  s, t_plus_1] > 0.0001)):

                        c1 = interp_color((1, 0, 0, 1), (0, 1, 0, 0), (self.deposits[s+1, t_plus_1]-min_dep)/rng_dep)
                        c2 = interp_color((1, 0, 0, 1), (0, 1, 0, 0), (self.deposits[s+1, t]-min_dep)/rng_dep)
                        c3 = interp_color((1, 0, 0, 1), (0, 1, 0, 0), (self.deposits[s, t_plus_1]-min_dep)/rng_dep)

                        glColor4f( *c1 )
                        glNormal3dv( self.deposit_norms[s+1, t_plus_1, :] )
                        glVertex( self.deposit_verts[s+1, t_plus_1, :] )
                        glColor4f( *c2 )
                        glNormal3dv( self.deposit_norms[s+1, t, :] )
                        glVertex( self.deposit_verts[s+1, t, :] )
                        glColor4f( *c3 )
                        glNormal3dv( self.deposit_norms[s, t_plus_1, :] )
                        glVertex( self.deposit_verts[s, t_plus_1, :] )

            glEnd()
    
            glEndList()

        else:
            
            # Render the display list            
            glCallList(self.display_list_deposit)

def run(terrain_seed=None):
    
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE, HWSURFACE|OPENGL|DOUBLEBUF)
    pygame.display.set_caption("PipeSim")
    
    resize(*SCREEN_SIZE)
    init()
    
    clock = pygame.time.Clock()    
    
    glMaterial(GL_FRONT, GL_AMBIENT, (0.1, 0.1, 0.1, 1.0))    
    glMaterial(GL_FRONT, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))

    # This object renders the pipe
    pipe = Pipe(60, in2m(30/2.0), deposit_seed=terrain_seed)

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


if __name__ == "__main__":

    if len(sys.argv) == 2:
        terrain_seed = int(sys.argv[1])
        run(terrain_seed)
    else:
        run()
