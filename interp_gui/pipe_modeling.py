import numpy as np
import math
from numpy.linalg import eig, inv

from draw_funcs import *

def poly_oval(x0,y0, x1,y1, steps=20, rotation=0):
    """return an oval as coordinates suitable for create_polygon"""

    # x0,y0,x1,y1 are as create_oval

    # rotation is in degrees anti-clockwise, convert to radians
    rotation = rotation * math.pi / 180.0

    # major and minor axes
    a = (x1 - x0) / 2.0
    b = (y1 - y0) / 2.0

    # center
    xc = x0 + a
    yc = y0 + b

    point_list = []

    # create the oval as a list of points
    for i in range(steps):

        # Calculate the angle for this step
        # 360 degrees == 2 pi radians
        theta = (math.pi * 2) * (float(i) / steps)

        x1 = a * math.cos(theta)
        y1 = b * math.sin(theta)

        # rotate x, y
        x = (x1 * math.cos(rotation)) + (y1 * math.sin(rotation))
        y = (y1 * math.cos(rotation)) - (x1 * math.sin(rotation))

        point_list.append(round(x + xc))
        point_list.append(round(y + yc))

    return np.array([point_list]).reshape((len(point_list), ))


def shannon_interp(theta, samples):
    W = 3*samples.size
    N = samples.size
    T = 2*np.pi/N
    
    if theta % T < T/2.:
        nearest_sample = int(theta - (theta % T))
    else:
        nearest_sample = (int(theta - (theta % T)) + 1) % N

    indices = range(nearest_sample-W+1, nearest_sample) + range(nearest_sample, nearest_sample+W)
    mod_indices = [i%N for i in indices]

    sum = 0
    iteration = 0
    for i in indices:
        sum += samples[mod_indices[iteration]]*np.sinc((theta-i*T)/T)
        iteration += 1
    return sum

   
# Inputs:
# canv  The canvas to which the pipe will be rendered
# meas  The measured values of the pipe radius 
# var   The variance assigned to each of the sensor readings in meas
# tag   The tag which will be used to label the pipe wall when it is drawn to the canvas

def render_pipe(canv, meas, var, tag):

    # Model the pipe for real and render it
    canv.delete(tag)

    prev_pt = None
    this_pt = None
    for theta in np.linspace(0, 2*np.pi, 100, endpoint=False):
        
        if prev_pt is None:
            interp_m = shannon_interp(theta, meas)
            first_pt = (world2screen_x(canv, interp_m*np.cos(theta)),
                        world2screen_y(canv, interp_m*np.sin(theta)))
            prev_pt = first_pt
        else:
            interp_m = shannon_interp(theta, meas)
            this_pt = (world2screen_x(canv, interp_m*np.cos(theta)),
                       world2screen_y(canv, interp_m*np.sin(theta)))

            interp_v = shannon_interp(theta, var)
            interp_v = min(1, max(interp_v, 0))
            redness = '{:02X}'.format(int(interp_v*255))
            blueness = '{:02X}'.format(int((1-interp_v)*255))
            color = '#'+redness+'00'+blueness

            line = (prev_pt[0], prev_pt[1], this_pt[0], this_pt[1])
            canv.create_line(line, fill=color, width=3, tag=tag)
            prev_pt = this_pt

    interp_v = shannon_interp(2*np.pi, var)
    redness = '{:02X}'.format(int(interp_v*255))
    blueness = '{:02X}'.format(int((1-interp_v)*255))
    color = '#'+redness+'00'+blueness
    line = (this_pt[0], this_pt[1], first_pt[0], first_pt[1])
    canv.create_line(line, fill=color, width=3, tag=tag)
