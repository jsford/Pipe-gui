import numpy as np
import math
from numpy.linalg import eig, inv


def model_pipe_slice(meas):
    coeffs = fitEllipse(meas[:,0], meas[:,1])
    center = ellipse_center(coeffs)
    rot = ellipse_angle_of_rotation(coeffs)
    lengths = ellipse_axis_length(coeffs)
    
    p0 = center[0]+lengths[0], center[1]-lengths[1]
    p1 = center[0]-lengths[0], center[1]+lengths[1]
    
    ovaltine = poly_oval(p0[0], p0[1], p1[0], p1[1], rotation=rot)
    
    return ovaltine


def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

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
    W = 4*samples.size
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

    if np.isnan(sum):
        print "OH NO"
        import pdb
        pdb.set_trace()
    return sum

    
