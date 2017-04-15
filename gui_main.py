#!/usr/bin/python

# Author: Jordan Ford
# Date: April 2017
# This program is an interactive GUI demonstrating the 
# differences between Shannon Interpolation and
# Least-Squares Interpolation as they apply to the 
# PipeDream pipe inspection project.

from Tkinter import *
import tkMessageBox as messagebox
import numpy as np
from math import *
import time
import random
import sys

from draw_funcs import *
from pipe_modeling import *

NUM_SENSORS = 8 
PIPE_DIAM_IN = 30


class PIPE_gui:
    def __init__(self, width, height):
        # Init the TK Window
        self.top = Tk()
        self.top.title("PipeDream Modeling Console")
        self.width = width
        self.height = height

        win_size_str = str(width) + "x" + str(height)
        self.top.geometry(win_size_str)
        
        self.top.configure(background=CHARCOAL)

        # Add Canvas
        self.canvas = Canvas(self.top, width=self.width-20, height=self.height-20,
                             background=LIGHT_GREY, borderwidth=0, highlightthickness=0)

        self.canvas.grid(row=0, column=0, columnspan=1, rowspan=1,
                         pady=(10, 10), padx=(10,10), sticky='nsew')
        

        self.canvas.bind("<ButtonPress-1>", self.move_start)
        self.canvas.bind("<B1-Motion>",     self.move_move)
        self.canvas.bind("<Button-4>",      self.zoomerM)
        self.canvas.bind("<Button-5>",      self.zoomerP)

        self.canvas.origin = (width/2.-10, height/2.-10)
        self.canvas.zl = 4.0

        # Handle resizing
        self.top.grid_rowconfigure(0, weight=1)
        self.top.grid_columnconfigure(0, weight=1)

        # Initialize the measurements
        
        self.measurements = np.zeros((NUM_SENSORS,))
        self.variances = np.zeros((NUM_SENSORS,))
        
        self.pipe_rad = in2cm(PIPE_DIAM_IN)/2.
        for i in range(0, NUM_SENSORS):
            self.measurements[i] = self.pipe_rad
            self.variances[i] = np.random.random()
        self.active_pt = 0

        # Bind arrow keys
        self.top.bind("<Left>",  self.on_left_press)
        self.top.bind("<Right>", self.on_right_press)
        self.top.bind("<Up>", self.on_up_press)
        self.top.bind("<Down>", self.on_down_press)
        self.top.bind("<Key>", self.on_key_press)


    # Run the tk mainloop.
    def mainloop(self):
        self.top.after(0, self.execute)
        self.top.mainloop();

    # Calls the planner and reschedules itself
    def execute(self):
        tic = time.clock()
        toc = time.clock()
    
        # Model the ellipse here. Pass the result to render()
        self.render()

        self.top.after(20, self.execute)

    # Is called by the planner to draw the world 
    def render(self):

        # Fit a least-squares ellipse to the pipe and render it
        m_pts = np.zeros((self.measurements.size, 2))

        i = 0
        for t in np.linspace(0, 2*pi, self.measurements.size, endpoint=False):
            m = self.measurements[i]
            m_pts[i,0] = m*cos(t)
            m_pts[i,1] = m*sin(t)
            i += 1

        oval_pts = model_pipe_slice(m_pts) 
        for o in range(0, oval_pts.size/2):
            oval_pts[2*o] = world2screen_x(self.canvas, oval_pts[2*o])
            oval_pts[2*o+1] = world2screen_y(self.canvas, oval_pts[2*o+1])
        
        self.canvas.delete('least_square_ellipse')
        self.canvas.create_polygon(tuple(oval_pts), outline=PASTEL_RED,
                                                    fill='', smooth=True,
                                                    width=3, dash=10,
                                                    tag='least_square_ellipse')

        # Model the pipe for real and render it
        self.canvas.delete('shannon_interp')

        prev_pt = None
        this_pt = None
        for theta in np.linspace(0, 2*pi, 100, endpoint=False):
            
            if prev_pt is None:
                interp_m = shannon_interp(theta, self.measurements)
                first_pt = (world2screen_x(self.canvas, interp_m*cos(theta)),
                           world2screen_y(self.canvas, interp_m*sin(theta)))
                prev_pt = first_pt
            else:
                interp_m = shannon_interp(theta, self.measurements)
                this_pt = (world2screen_x(self.canvas, interp_m*cos(theta)),
                           world2screen_y(self.canvas, interp_m*sin(theta)))

                interp_v = shannon_interp(theta, self.variances)
                interp_v = min(1, max(interp_v, 0))
                redness = '{:02X}'.format(int(interp_v*255))
                blueness = '{:02X}'.format(int((1-interp_v)*255))
                color = '#'+redness+'00'+blueness

                line = (prev_pt[0], prev_pt[1], this_pt[0], this_pt[1])
                self.canvas.create_line(line, fill=color, width=3, tag='shannon_interp')
                prev_pt = this_pt

        interp_v = shannon_interp(2*pi, self.variances)
        redness = '{:02X}'.format(int(interp_v*255))
        blueness = '{:02X}'.format(int((1-interp_v)*255))
        color = '#'+redness+'00'+blueness
        line = (this_pt[0], this_pt[1], first_pt[0], first_pt[1])
        self.canvas.create_line(line, fill=color, width=3, tag='shannon_interp')

        # Render the measurements points
        self.canvas.delete('m_pt')

        for m in range(0, self.measurements.size):
            p1 = (world2screen_x(self.canvas, m_pts[m,0]-2),
                  world2screen_y(self.canvas, m_pts[m,1]-2))
            p2 = (world2screen_x(self.canvas, m_pts[m,0]+2),
                  world2screen_y(self.canvas, m_pts[m,1]+2))

            v = self.variances[m]
            redness = '{:02X}'.format(int(v*255))
            blueness = '{:02X}'.format(int((1-v)*255))
            color = '#'+redness+'00'+blueness

            if m == self.active_pt:
                self.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], 
                                        fill=color, outline=PASTEL_GREEN, tag='m_pt', width=3)
            else:
                self.canvas.create_oval(p1[0], p1[1], p2[0], p2[1], 
                                        fill=color, outline=PASTEL_YELLOW, tag='m_pt', width=3)


        # Display crosshairs at the origin
        self.canvas.delete('origin')
        c1 = (world2screen_x(self.canvas, 0), world2screen_y(self.canvas,  10))
        c2 = (world2screen_x(self.canvas, 0), world2screen_y(self.canvas, -10))
        self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill='white', tag='origin')
        c1 = (world2screen_x(self.canvas,  10), world2screen_y(self.canvas, 0))
        c2 = (world2screen_x(self.canvas, -10), world2screen_y(self.canvas, 0))
        self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill='white', tag='origin')
        
        

    # Click and drag the canvas using the mouse
    def move_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
    def move_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # Update the current mouse coordinates on the canvas
    def mouse_move(self, event):
        mx = screen2world_x(self.canvas, self.canvas.canvasx(event.x))
        my = screen2world_y(self.canvas, self.canvas.canvasy(event.y))

        self.mouse_coord_disp.delete('1.0', END)
        self.mouse_coord_disp.insert('1.0', "("+format(pix2m(mx),'.2f')+", "
                                               +format(pix2m(my),'.2f')+")", "STYLE")
        self.mouse_coord_disp.tag_config("STYLE", foreground='white', justify='right')
        

    # Zoom using mouse scrollwheel 
    def zoomerP(self,event):
        MAX_ZOOM = 50
        if (self.canvas.zl >= MAX_ZOOM): self.canvas.zl = MAX_ZOOM; return;

        (mx, my) = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        # Move the origin to a new point calculated from the zoom center
        self.canvas.origin = (self.canvas.origin[0] - (mx-self.canvas.origin[0])*0.1,
                              self.canvas.origin[1] - (my-self.canvas.origin[1])*0.1)

        self.canvas.old_zl = self.canvas.zl
        self.canvas.zl *= 1.1

    def zoomerM(self,event):
        MIN_ZOOM = 0.001 
        if (self.canvas.zl <= MIN_ZOOM): self.canvas.zl = MIN_ZOOM; return;

        (mx, my) = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        # Move the origin to a new point calculated from the zoom center
        self.canvas.origin = (mx - (mx-self.canvas.origin[0])/1.1,
                              my - (my-self.canvas.origin[1])/1.1)

        self.canvas.old_zl = self.canvas.zl
        self.canvas.zl /= 1.1 
    
    def on_left_press(self, event):
        self.active_pt += 1
        if self.active_pt >= self.measurements.size: 
            self.active_pt = 0

    def on_right_press(self, event):
        self.active_pt -= 1
        if self.active_pt < 0:
             self.active_pt = self.measurements.size-1

    def on_up_press(self, event):
        self.measurements[self.active_pt] /= 0.98

    def on_down_press(self, event):
        self.measurements[self.active_pt] *= 0.98

    def on_key_press(self, event):
        if event.char == 'r' or event.char == 'R':
            for i in range(0, NUM_SENSORS):
                self.measurements[i] = self.pipe_rad
            self.active_pt = 0


if __name__ == "__main__":

    print "\n\n"
    print "Usage:"
    print "\tLeft Arrow Key moves clockwise."
    print "\tRight Arrow Key moves counter-clockwise."
    print "\tUp Arrow Key moves control point away from the center."
    print "\tDown Arrow Key moves control point toward the center."
    print ""
    print "\tRed dots represent high variance sensor readings."
    print "\tBlue dots represent low variance sensor reading."
    print ""
    print "\tThe dashed line is the least-squares ellipse fit to the control points."
    print "\tThe red/blue line is the shannon interpolated line color-coded by uncertainty."
    

    pipedream_gui = PIPE_gui(960, 700)
    pipedream_gui.mainloop() 

