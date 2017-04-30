#!/usr/bin/python
SCREEN_SIZE = (800, 600)

import numpy as np
import scipy.io
from math import radians 
import colorsys
import sys
import shutil
import argparse

from vector3 import *

from PIL import Image
from collada import *

from terrain.generator import *

def ft2m(feet):
    return feet*0.3048
def in2m(inches):
    return inches*0.0254


class Pipe(object):
    def __init__(self, pipe_length_m=61, pipe_radius_m=in2m(15),                        
                 max_deposit_thickness_mm=25.4, uniform=False,                          
                 ovality=1.0, dirtiness=0.9, deposit_seed=None,                         
                 output_filename='test'):

        self.pipe_length_m = pipe_length_m
        self.pipe_rad_m = pipe_radius_m
        self.max_deposit_thickness_mm = max_deposit_thickness_mm
        self.uniform = uniform
        self.ovality = ovality
        self.dirtiness = dirtiness
        self.output_filename=output_filename
        self.deposit_seed = deposit_seed

        # Use ~64 samples/m radially 
        # Use ~256 samples/10m longitudinally 
        self.rad_samples = 2**int(np.log2(64*np.pi*2*self.pipe_rad_m))     # Radial samples
        self.lon_samples = 2**int(np.log2(25.6*self.pipe_length_m))        # Longitudinal samples

        # Create the deposit surface using the
        # diamond-square terrain generation algorithm.
        # Use the random seed and cleanliness threshold provided.
        # Use the sampling density derived above.
        print 'Generating deposit surface grid with dimension ' + str((self.rad_samples, self.lon_samples))
        if self.uniform:
            self.deposits = np.ones((self.lon_samples, self.rad_samples))
        else: 
            self.deposits = DepositGenerator(self.rad_samples, self.lon_samples, threshold=(1-dirtiness), seed=self.deposit_seed).pipe_map.transpose()
        self.deposits *= max_deposit_thickness_mm/1000.0 

        self.pipe_rad = np.empty_like(self.deposits)
        self.pipe_verts = np.zeros((self.lon_samples, self.rad_samples, 3))
        self.pipe_norms = np.zeros((self.lon_samples, self.rad_samples, 3))

        
        self.deposit_verts = np.zeros((self.lon_samples, self.rad_samples, 3))
        self.deposit_norms = np.zeros((self.lon_samples, self.rad_samples, 3))

        self.pipe_axis = Vector3.from_floats(0, 0, -1)

        for s in range(0, self.lon_samples):
            center = tuple(self.pipe_axis*(s/float(self.lon_samples)*self.pipe_length_m) + Vector3(0,0,0))
            for t in range(0, self.rad_samples):
                theta = t*2*np.pi/self.rad_samples
                self.pipe_rad[s,t] = self.ovality*self.pipe_rad_m / np.sqrt(self.ovality**2+1)
                self.pipe_verts[s,t,:] = (center[0]+self.ovality*self.pipe_rad_m*np.cos(theta), center[1]+self.pipe_rad_m*np.sin(theta), center[2])
                self.pipe_norms[s,t,:] = (tuple(Vector3.from_points(self.pipe_verts[s,t,:], center))) 
                self.pipe_norms[s,t,:] /= np.linalg.norm(self.pipe_norms[s,t,:])
                r = self.pipe_rad_m - self.deposits[s,t]
                self.deposit_verts[s,t] = (center[0]+r*self.ovality*np.cos(theta), center[1]+r*np.sin(theta), center[2])

        self.deposit_norms = self.pipe_norms

        self.export_pipe_description()
        self.export_texture()
        self.export_pipe()
        self.export_world()
        self.export_mat()
        self.export_npz()

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

    def export_pipe_description(self):
        f = open(self.output_filename+'/pipe.params', 'w')
        f.write('Pipe Description File\n')
        f.write('Author: Jordan Ford\n')
        f.write('This file lists the parameters needed to recreate a pipe.\n\n')
        f.write('Description: ' + self.output_filename + '\n')
        f.write('\n')
        f.write('PIPE\n')
        f.write('-----------------------------------------\n')
        f.write('Pipe Length [m]: ' + str(self.pipe_length_m) + '\n')
        f.write('Pipe Radius [m]: ' + str(self.pipe_rad_m) + '\n')
        f.write('\n')
        f.write('DEPOSIT\n')
        f.write('-----------------------------------------\n')
        f.write('Deposit RNG seed: ' + str(self.deposit_seed) + '\n')
        f.write('Max. Deposit Thickness [mm]: ' + str(self.max_deposit_thickness_mm) + '\n')
        f.write('Dirtiness [%]: ' + str(100*self.dirtiness) + '\n')
        f.close()

    def export_texture(self):
        print 'Exporting deposit surface to ' + self.output_filename + '/texture.png'
        min_dep = 0
        max_dep = np.amax(self.deposits)
        rng_dep = max_dep - min_dep

        deposit_colors = np.zeros((self.lon_samples, self.rad_samples, 4), dtype=np.uint8)
        for s in range(0, self.lon_samples):
            for t in range(0, self.rad_samples):
                deposit_colors[s,t,:] = self.interp_color((1,0,0,1), (0,1,0,0),
                                                        (self.deposits[s, t]-min_dep)/rng_dep)
        texture = Image.fromarray(deposit_colors,'RGBA')
        texture.save(self.output_filename+'/texture.png')

    def export_world(self):
        print 'Exporting world to ' + self.output_filename + '/pipe.world'

        if self.output_filename == 'template.world':
            print 'ERROR: output filename cannot be \'template\''
            exit(1)

        template_handle = open('template.world', 'r')
        world_file_str = template_handle.read()
        world_file_str = world_file_str.replace('XXXXXX', self.output_filename+'/pipe.dae')
        world_file_handle = open(self.output_filename+'/pipe.world', 'w')
        world_file_handle.write(world_file_str)
        world_file_handle.close()

    def export_pipe(self):
        print 'Exporting pipe to ' + self.output_filename + '/pipe.dae'

        mesh = Collada()
        mesh.assetInfo.upaxis = asset.UP_AXIS.Z_UP

        # Create the pipe node

        pipe_effect = material.Effect('pipe_effect', [], 'phong', diffuse=(.91,.91,.91), specular=(.1,.1,.1), ambient=(.91, .91, .91), double_sided=True)
        pipe_mat = material.Material('pipe_material', 'nickel', pipe_effect)
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
        for s in range(0, self.lon_samples-1):
            for t in range(0, self.rad_samples):
                t_plus_one = (t+1)%self.rad_samples

                pipe_indices.append(s*self.rad_samples+t)                    # v1
                pipe_indices.append(s*self.rad_samples+t)                    # n1
                pipe_indices.append(s*self.rad_samples+t_plus_one)           # v2
                pipe_indices.append(s*self.rad_samples+t_plus_one)           # n2
                pipe_indices.append((s+1)*self.rad_samples+t)                # v3
                pipe_indices.append((s+1)*self.rad_samples+t)                # n3

                pipe_indices.append(s*self.rad_samples+t_plus_one)           # v1
                pipe_indices.append(s*self.rad_samples+t_plus_one)           # n1
                pipe_indices.append((s+1)*self.rad_samples+t_plus_one)       # v2
                pipe_indices.append((s+1)*self.rad_samples+t_plus_one)       # n2
                pipe_indices.append((s+1)*self.rad_samples+t)                # v3
                pipe_indices.append((s+1)*self.rad_samples+t)                # n3
        

        pipe_triset = pipe_geom.createTriangleSet(np.array(pipe_indices), pipe_input_list, 'materialref')
        pipe_geom.primitives.append(pipe_triset)
        mesh.geometries.append(pipe_geom)

        pipe_matnode = scene.MaterialNode('materialref', pipe_mat, inputs=[])
        pipe_geomnode = scene.GeometryNode(pipe_geom, [pipe_matnode])
        pipe_node = scene.Node('node0', children=[pipe_geomnode])


        # Create the deposit node

        image = material.CImage('deposit_texture', self.output_filename+'/texture.png')
        surface = material.Surface('deposit_surface', image)
        sampler2d = material.Sampler2D('deposit_sampler', surface)
        tex_map = material.Map(sampler2d, 'UVSET0')
        dep_effect = material.Effect('deposit_effect', [surface, sampler2d], 'phong', emission=(0,0,0,1),\
                                     ambient=(1,1,1,1), diffuse=tex_map,\
                                     transparent=tex_map, transparency=0.0, double_sided=True)

        dep_mat = material.Material('deposit_material_ID', 'uranylFluoride', dep_effect)
        mesh.effects.append(dep_effect)
        mesh.materials.append(dep_mat)
        mesh.images.append(image)

        dep_uv = []
        for s in range(0, self.lon_samples):
            for t in range(0, self.rad_samples):
                dep_uv.append(t/float(self.rad_samples-1))
                dep_uv.append(1-s/float(self.lon_samples-1))


        dep_indices = [] 
        for s in range(0, self.lon_samples-1):
            for t in range(0, self.rad_samples):
                t_plus_one = (t+1)%self.rad_samples

                dep_indices.append(s*self.rad_samples+t)                     # v1
                dep_indices.append(s*self.rad_samples+t)                     # n1
                dep_indices.append(s*self.rad_samples+t)                     # uv1
                dep_indices.append(s*self.rad_samples+t_plus_one)            # v2
                dep_indices.append(s*self.rad_samples+t_plus_one)            # n2
                dep_indices.append(s*self.rad_samples+t_plus_one)            # uv2
                dep_indices.append((s+1)*self.rad_samples+t)                 # v3
                dep_indices.append((s+1)*self.rad_samples+t)                 # n3
                dep_indices.append((s+1)*self.rad_samples+t)                 # uv3

                dep_indices.append(s*self.rad_samples+t_plus_one)            # v1
                dep_indices.append(s*self.rad_samples+t_plus_one)            # n1
                dep_indices.append(s*self.rad_samples+t_plus_one)            # uv1
                dep_indices.append((s+1)*self.rad_samples+t_plus_one)        # v2
                dep_indices.append((s+1)*self.rad_samples+t_plus_one)        # n2
                dep_indices.append((s+1)*self.rad_samples+t_plus_one)        # uv2
                dep_indices.append((s+1)*self.rad_samples+t)                 # v3
                dep_indices.append((s+1)*self.rad_samples+t)                 # n3
                dep_indices.append((s+1)*self.rad_samples+t)                 # uv3

        
        dep_v = np.reshape(self.deposit_verts, self.pipe_verts.size)
        dep_n = np.reshape(self.deposit_norms, self.pipe_norms.size)
        dep_vert_src = source.FloatSource('depverts-array', dep_v, ('X','Y','Z'))
        dep_norm_src = source.FloatSource('depnorms-array', dep_n, ('X','Y','Z'))
        dep_uv_src = source.FloatSource('depuv-array', np.array(dep_uv), ('S', 'T'))

        dep_geom = geometry.Geometry(mesh, 'geometry1', 'deposit', [dep_vert_src, dep_norm_src, dep_uv_src])
        dep_input_list = source.InputList()
        dep_input_list.addInput(0, 'VERTEX', '#depverts-array')
        dep_input_list.addInput(1, 'NORMAL', '#depnorms-array')
        dep_input_list.addInput(2, 'TEXCOORD', '#depuv-array', set='0')

        dep_triset = dep_geom.createTriangleSet(np.array(dep_indices), dep_input_list, 'materialref')
        dep_geom.primitives.append(dep_triset)
        mesh.geometries.append(dep_geom)

        dep_matnode = scene.MaterialNode('materialref', dep_mat, inputs=[])
        dep_geomnode = scene.GeometryNode(dep_geom, [dep_matnode])
        dep_node = scene.Node('node0', children=[dep_geomnode])

        myscene = scene.Scene('myscene', [pipe_node, dep_node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        mesh.write(self.output_filename+'/pipe.dae')

    def export_mat(self):
        print 'Exporting pipe and deposit to .mat file: ' + self.output_filename + '/pipe.mat'
        dict = {'deposit':self.deposits,
                'pipe_rad':self.pipe_rad,
                'pipe_verts':self.pipe_verts,
                'pipe_norms':self.pipe_norms}
        scipy.io.savemat(self.output_filename+'/pipe.mat', dict, appendmat=False)

    def export_npz(self):
        print 'Exporting deposit to .npz file: ' + self.output_filename + '/pipe.npz'
        np.savez(self.output_filename+'/pipe', deposit=self.deposits,
                                              pipe_rad=self.pipe_rad,
                                              pipe_verts=self.pipe_verts,
                                              pipe_norms=self.pipe_norms) 
        
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius_m', help='The radius of the pipe in meters.',
                        type=float, default=in2m(15))
    parser.add_argument('-l', '--length_m', help='The length of the pipe in meters.',
                        type=float, default=61)
    parser.add_argument('-t', '--thickness_mm', help='The maximum thickness of deposit.',
                        type=float, default=25.4)
    parser.add_argument('-d', '--dirtiness', help='The dirtiness of the pipe as a percent.',
                        type=float, default=40)
    parser.add_argument('-u', '--uniform', help='Specify that the deposit be uniformly distributed.',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='The prefix used for all output files.',
                        type=str, default='test')
    parser.add_argument('-s', '--seed', help='An integer seed used by the RNG to generate the deposit.',
                        type=int, default=None)
    parser.add_argument('-oval', '--ovality', help='The percent ovality of the pipe. > 1 is wider. < 1 is taller.',
                        type=float, default=1.0)
    args = parser.parse_args()

    if os.path.exists(args.output):
        print 'Overwriting output directory: ./' + args.output
        shutil.rmtree(args.output)
        os.mkdir(args.output, 0755);
    else:
        print 'Creating output directory: ./' + args.output
        os.mkdir(args.output, 0755);
        
    if args.length_m <= args.radius_m:
        print 'ERROR: Aspect ratio is too extreme make your pipe longer or reduce its diameter.'
        exit(1)

    pipe = Pipe(pipe_length_m                =          args.length_m,
                pipe_radius_m                =          args.radius_m,
                max_deposit_thickness_mm     =          args.thickness_mm,
                uniform                      =          args.uniform,
                ovality                      =          args.ovality, 
                dirtiness                    =          args.dirtiness/100.0,
                deposit_seed                 =          args.seed,
                output_filename              =          args.output)













