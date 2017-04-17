#! /usr/bin/python

import rosbag
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
import numpy as np
import tf.transformations

bag = rosbag.Bag('bags/ak_pipe_test1_30_4.bag')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pos_valid = False
in_pipe = False

pos_x = []
pos_y = []
pos_z = []

cx = []
cy = []
cz = []

roll = []
pitch = []
yaw = []


for topic, msg, _t in bag.read_messages(['/ak1/odometry/filtered', '/ak1/scan']):

    if topic == '/ak1/odometry/filtered':
        pos_valid = True
        cx.append(msg.pose.pose.position.x)
        cy.append(msg.pose.pose.position.y)
        cz.append(msg.pose.pose.position.z)

        quat = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quat)
        roll.append(euler[0])
        pitch.append(euler[1])
        yaw.append(euler[2])

    elif pos_valid and topic == '/ak1/scan':

        if in_pipe == False:
            in_pipe = np.sum(msg.intensities)/(47*360) > 0.8
            
        if in_pipe == True: 
            i = 0
            for d in np.linspace(0, 2*np.pi, 360, endpoint=False):
                if msg.ranges[i] != float('inf'):
                    r = msg.ranges[i]
                    if r < 5:
                        pos_x.append(np.sin(yaw[-1])*r + cx[-1])
                        pos_y.append(np.sin(d)*np.cos(yaw[-1])*(r))
                        pos_z.append(np.cos(d)*(r))
                else:
                    in_pipe = False
                i += 1 
            pos_valid = False

        
ax.plot(cx,cy,cz, zdir='y', color='red')
ax.plot(pos_x,pos_y,pos_z, zdir='y')
ax.set_xlim(0, 3)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.show()

plt.plot(pos_x, pos_y)
plt.plot(cx, cy, color='red')
plt.title("X-Y Proj.")
plt.show()

plt.plot(pos_x, pos_z)
plt.plot(cx, cz, color='red')
plt.title("X-Z Proj.")
plt.show()

plt.plot(pos_z, pos_y)
plt.plot(cz, cy, color='red')
plt.title("Z-Y Proj.")
plt.axes().set_aspect('equal','datalim')
plt.show()

plt.plot(cx, label='X')
plt.plot(cy, label='Y')
plt.plot(cz, label='Z')
plt.legend()
plt.show()

bag.close()

