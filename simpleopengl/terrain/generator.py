import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D


def _map_get(map, x, y):
    return map[x%(map.shape[0]), y%(map.shape[1])]

def _map_set(map, x, y, val):
    map[x%(map.shape[0]), y%(map.shape[1])] = val

def _diamond_avg(map, tile_x, tile_y, tile_size):
    diamond_sum  = _map_get(map, tile_x, tile_y)
    diamond_sum += _map_get(map, (tile_x+tile_size), tile_y)
    diamond_sum += _map_get(map, tile_x, (tile_y+tile_size))
    diamond_sum += _map_get(map, (tile_x+tile_size), (tile_y+tile_size))

    return diamond_sum/4.0


def _square_avg(map, tile_x, tile_y, tile_size):
    half_tile_size = tile_size/2
    square_sum  = _map_get(map, (tile_x-half_tile_size), tile_y)
    square_sum += _map_get(map, (tile_x+half_tile_size), tile_y)
    square_sum += _map_get(map, tile_x, (tile_y-half_tile_size))
    square_sum += _map_get(map, tile_x, (tile_y+half_tile_size))
   
    return square_sum/4.0 
        

def diamond_square_alg(scale=7, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Technically, tile size is 2**scale + 1, but this works out much
    # more conveniently since most of the time I need tile_size-1 not tile_size.
    tile_size = 2**scale
    map = np.zeros((tile_size, tile_size))

    # Initialize the corner values
    _map_set(map, 0, 0, 2*np.random.rand())
    _map_set(map, tile_size, 0, 2*np.random.rand())
    _map_set(map, 0, tile_size, 2*np.random.rand())
    _map_set(map, tile_size, tile_size, 2*np.random.rand())


    rand_scale = 1.0
    while scale > 0:
        tile_size = 2**scale
        half_tile_size = int(2**(scale-1))

        for tile_x in range(0, map.shape[0]-half_tile_size, tile_size):
            for tile_y in range(0, map.shape[1]-half_tile_size, tile_size):
                d_avg = _diamond_avg(map, tile_x, tile_y, tile_size)
                perturb = (2*np.random.rand()-1)*rand_scale
                _map_set(map, tile_x+half_tile_size, tile_y+half_tile_size, d_avg + perturb)

        for tile_x in range(0, map.shape[0], half_tile_size):
            if tile_x % (tile_size) == 0:
                y_range = range(half_tile_size, map.shape[1], tile_size)
            else:
                y_range = range(0, map.shape[1], tile_size)
            for tile_y in y_range:
                s_avg = _square_avg(map, tile_x, tile_y, tile_size)
                perturb = (2*np.random.rand()-1)*rand_scale
                _map_set(map, tile_x, tile_y, s_avg + perturb)

        scale -= 1
        rand_scale /= 2.0

        map -= np.amin(map)
        map /= np.amax(map)
        
    return map


def threshold_map(map, threshold):
    map[map < threshold] = threshold
    map -= threshold
    map /= (1-threshold)
    return map


if __name__ == "__main__":

    Z = diamond_square_alg(scale=4)
    
    X = np.linspace(0, Z.shape[0], Z.shape[0], endpoint=False)
    Y = np.linspace(0, Z.shape[1], Z.shape[1], endpoint=False)
    (X, Y) = np.meshgrid(X,Y)


    fig = plt.figure()
    ax  = Axes3D(fig)
    ax.plot_surface(X, Y, Z.transpose(), rstride=1, cstride=1, cmap=cm.coolwarm) 
    plt.show()
    






