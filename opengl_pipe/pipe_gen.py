#!/usr/bin/python
SCREEN_SIZE = (800, 600)

import numpy as np
from math import radians 
import colorsys
import sys

from gameobjects.vector3 import *

from PIL import Image
from collada import *

from terrain.generator import *

def ft2m(feet):
    return feet*0.3048
def in2m(inches):
    return inches*0.0254



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
        
        self.deposits = DepositGenerator(128, 256, threshold=0.2).pipe_map.transpose()
        self.deposits *= in2m(4.0) 

        self.pipe_verts = np.zeros((256, 128, 3))
        self.pipe_norms = np.zeros((256, 128, 3))

        self.deposit_verts = np.zeros((256, 128, 3))
        self.deposit_norms = np.zeros((256, 128, 3))

        for s in range(0, 256):
            center = tuple(self.axis*(s/256.0*self.length) + Vector3(0,0,0))
            for t in range(0, 128):
                theta = t*2*np.pi/128.0
                self.pipe_verts[s,t,:] = (center[0]+self.rad*np.cos(theta), center[1]+self.rad*np.sin(theta), center[2])
                self.pipe_norms[s,t,:] = (tuple(Vector3.from_points(self.pipe_verts[s,t,:], center))) 
                self.pipe_norms[s,t,:] /= np.linalg.norm(self.pipe_norms[s,t,:])
                r = self.rad - self.deposits[s,t]
                self.deposit_verts[s,t] = (center[0]+r*np.cos(theta), center[1]+r*np.sin(theta), center[2])

        self.deposit_norms = self.pipe_norms

        self.generate_texture()
        self.export_pipe()

    def generate_texture(self):
        min_dep = 0
        max_dep = np.amax(self.deposits)
        rng_dep = max_dep - min_dep

        deposit_colors = np.zeros((256, 128, 4), dtype=np.uint8)
        for s in range(0, 256):
            for t in range(0, 128):
                deposit_colors[s,t,:] = self.interp_color((1,0,0,1), (0,1,0,0),
                                                        (self.deposits[s, t]-min_dep)/rng_dep)
        texture = Image.fromarray(deposit_colors,'RGBA')
        texture.save('deposit.png')

    def export_pipe(self):
        mesh = Collada()
        mesh.assetInfo.upaxis = asset.UP_AXIS.Z_UP

        # Create the pipe node

        pipe_effect = material.Effect("pipe_effect", [], "phong", diffuse=(.91,.91,.91), specular=(.1,.1,.1), ambient=(.91, .91, .91), double_sided=True)
        pipe_mat = material.Material("pipe_material", "nickel", pipe_effect)
        mesh.effects.append(pipe_effect)
        mesh.materials.append(pipe_mat)

        pipe_v = np.reshape(self.pipe_verts, self.pipe_verts.size)
        pipe_n = np.reshape(self.pipe_norms, self.pipe_norms.size)
        pipe_vert_src = source.FloatSource('pipeverts-array', pipe_v, ('X','Y','Z'))
        pipe_norm_src = source.FloatSource('pipenorms-array', pipe_n, ('X','Y','Z'))

        pipe_geom = geometry.Geometry(mesh, 'geometry0', 'pipe', [pipe_vert_src, pipe_norm_src])
        pipe_input_list = source.InputList()
        pipe_input_list.addInput(0, 'VERTEX', '#pipeverts-array')
        pipe_input_list.addInput(1, 'NORMAL', '#pipenorms-array')

        pipe_indices = [] 
        for s in range(0, 256-1):
            for t in range(0, 128):
                t_plus_one = (t+1)%128

                pipe_indices.append(s*128+t)                    # v1
                pipe_indices.append(s*128+t)                    # n1
                pipe_indices.append(s*128+t_plus_one)           # v2
                pipe_indices.append(s*128+t_plus_one)           # n2
                pipe_indices.append((s+1)*128+t)                # v3
                pipe_indices.append((s+1)*128+t)                # n3

                pipe_indices.append(s*128+t_plus_one)           # v1
                pipe_indices.append(s*128+t_plus_one)           # n1
                pipe_indices.append((s+1)*128+t_plus_one)       # v2
                pipe_indices.append((s+1)*128+t_plus_one)       # n2
                pipe_indices.append((s+1)*128+t)                # v3
                pipe_indices.append((s+1)*128+t)                # n3
        

        pipe_triset = pipe_geom.createTriangleSet(np.array(pipe_indices), pipe_input_list, 'materialref')
        pipe_geom.primitives.append(pipe_triset)
        mesh.geometries.append(pipe_geom)

        pipe_matnode = scene.MaterialNode("materialref", pipe_mat, inputs=[])
        pipe_geomnode = scene.GeometryNode(pipe_geom, [pipe_matnode])
        pipe_node = scene.Node('node0', children=[pipe_geomnode])


        # Create the deposit node

        image = material.CImage('deposit_texture', 'deposit.png')
        surface = material.Surface('deposit_surface', image)
        sampler2d = material.Sampler2D('deposit_sampler', surface)
        tex_map = material.Map(sampler2d, 'UVSET0')
        dep_effect = material.Effect('deposit_effect', [surface, sampler2d], 'lambert', emission=(0,0,0,1),\
                                     ambient=(1,1,1,1), diffuse=tex_map,\
                                     transparent=tex_map, transparency=0.0, double_sided=True)

        dep_mat = material.Material("deposit_material_ID", "uranylFluoride", dep_effect)
        mesh.effects.append(dep_effect)
        mesh.materials.append(dep_mat)
        mesh.images.append(image)

        dep_uv = []
        for s in range(0, 256):
            for t in range(0, 128):
                dep_uv.append(t/127.0)
                dep_uv.append(s/255.0)


        dep_indices = [] 
        for s in range(0, 256-1):
            for t in range(0, 128):
                t_plus_one = (t+1)%128

                dep_indices.append(s*128+t)                     # v1
                dep_indices.append(s*128+t)                     # n1
                dep_indices.append(s*128+t)                     # uv1
                dep_indices.append(s*128+t_plus_one)            # v2
                dep_indices.append(s*128+t_plus_one)            # n2
                dep_indices.append(s*128+t_plus_one)            # uv2
                dep_indices.append((s+1)*128+t)                 # v3
                dep_indices.append((s+1)*128+t)                 # n3
                dep_indices.append((s+1)*128+t)                 # uv3

                dep_indices.append(s*128+t_plus_one)            # v1
                dep_indices.append(s*128+t_plus_one)            # n1
                dep_indices.append(s*128+t_plus_one)            # uv1
                dep_indices.append((s+1)*128+t_plus_one)        # v2
                dep_indices.append((s+1)*128+t_plus_one)        # n2
                dep_indices.append((s+1)*128+t_plus_one)        # uv2
                dep_indices.append((s+1)*128+t)                 # v3
                dep_indices.append((s+1)*128+t)                 # n3
                dep_indices.append((s+1)*128+t)                 # uv3

        
        dep_v = np.reshape(self.deposit_verts, self.pipe_verts.size)
        dep_n = np.reshape(self.deposit_norms, self.pipe_norms.size)
        dep_vert_src = source.FloatSource('depverts-array', dep_v, ('X','Y','Z'))
        dep_norm_src = source.FloatSource('depnorms-array', dep_n, ('X','Y','Z'))
        dep_uv_src = source.FloatSource('depuv-array', np.array(dep_uv), ('S', 'T'))

        dep_geom = geometry.Geometry(mesh, 'geometry1', 'deposit', [dep_vert_src, dep_norm_src, dep_uv_src])
        dep_input_list = source.InputList()
        dep_input_list.addInput(0, 'VERTEX', '#depverts-array')
        dep_input_list.addInput(1, 'NORMAL', '#depnorms-array')
        dep_input_list.addInput(2, 'TEXCOORD', '#depuv-array', set="0")

        dep_triset = dep_geom.createTriangleSet(np.array(dep_indices), dep_input_list, 'materialref')
        dep_geom.primitives.append(dep_triset)
        mesh.geometries.append(dep_geom)

        dep_matnode = scene.MaterialNode("materialref", dep_mat, inputs=[])
        dep_geomnode = scene.GeometryNode(dep_geom, [dep_matnode])
        dep_node = scene.Node('node0', children=[dep_geomnode])

        # TODO: Add pipe_node back here
        myscene = scene.Scene('myscene', [dep_node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write('test.dae')
        

    def interp_color(self, c1, c2, t):

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

        return (255*r, 255*g, 255*b, 255*alpha)

if __name__ == "__main__":

    if len(sys.argv) == 2:
        terrain_seed = int(sys.argv[1])
        pipe = Pipe(10, in2m(30/2.0), deposit_seed=terrain_seed)
    else:
        pipe = Pipe(10, in2m(30/2.0), deposit_seed=None)
