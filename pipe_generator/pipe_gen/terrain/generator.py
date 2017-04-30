#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Issues:
# Fix discontinuities between pipe segments.
# Convert to C++ for speed and use in ROS sim

class DepositGenerator:

    def __init__(self, samples_radial, samples_longitudinal, threshold=None, seed=None):
        log_samples_radial = int(np.log2(samples_radial))
        log_samples_longitudinal = int(np.log2(samples_longitudinal))

        samples_radial = 2**log_samples_radial;
        samples_longitudinal = 2**log_samples_longitudinal;

        pipe_map = np.zeros((2**log_samples_radial, samples_longitudinal))
        
        num_sections = 2**(log_samples_longitudinal - log_samples_radial)

        section_longitude = 0
        for section_idx in range(0, num_sections):
            section = self.generate_pipe_section(scale=log_samples_radial, seed=seed)
            longitude_range = range(section_longitude, 
                                    section_longitude+section.shape[1])
            pipe_map[:, longitude_range] = section
            section_longitude += section.shape[1]

        self.pipe_map = pipe_map

        if threshold is not None:
            self._threshold_map(threshold)
            

    def _map_get(self, map, x, y):
        return map[x%(map.shape[0]), y%(map.shape[1])]

    def _map_set(self, map, x, y, val):
        map[x%(map.shape[0]), y%(map.shape[1])] = val

    def _diamond_avg(self, map, tile_x, tile_y, tile_size):
        diamond_sum  = self._map_get(map, tile_x, tile_y)
        diamond_sum += self._map_get(map, (tile_x+tile_size), tile_y)
        diamond_sum += self._map_get(map, tile_x, (tile_y+tile_size))
        diamond_sum += self._map_get(map, (tile_x+tile_size), (tile_y+tile_size))

        return diamond_sum/4.0


    def _square_avg(self, map, tile_x, tile_y, tile_size):
        half_tile_size = tile_size/2
        square_sum  = self._map_get(map, (tile_x-half_tile_size), tile_y)
        square_sum += self._map_get(map, (tile_x+half_tile_size), tile_y)
        square_sum += self._map_get(map, tile_x, (tile_y-half_tile_size))
        square_sum += self._map_get(map, tile_x, (tile_y+half_tile_size))
       
        return square_sum/4.0 
            

    # Uses the diamond square algorithm.
    def generate_pipe_section(self, scale=7, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Technically, tile size is 2**scale + 1, but this works out much
        # more conveniently since most of the time I need tile_size-1 not tile_size.
        tile_size = 2**scale
        map = np.zeros((tile_size, tile_size))

        # Initialize the corner values
        self._map_set(map, 0, 0, 2*np.random.rand())
        self._map_set(map, tile_size, 0, 2*np.random.rand())
        self._map_set(map, 0, tile_size, 2*np.random.rand())
        self._map_set(map, tile_size, tile_size, 2*np.random.rand())


        rand_scale = 1.0
        while scale > 0:
            tile_size = 2**scale
            half_tile_size = int(2**(scale-1))

            for tile_x in range(0, map.shape[0]-half_tile_size, tile_size):
                for tile_y in range(0, map.shape[1]-half_tile_size, tile_size):
                    d_avg = self._diamond_avg(map, tile_x, tile_y, tile_size)
                    perturb = (2*np.random.rand()-1)*rand_scale
                    self._map_set(map, tile_x+half_tile_size, tile_y+half_tile_size, d_avg + perturb)

            for tile_x in range(0, map.shape[0], half_tile_size):
                if tile_x % (tile_size) == 0:
                    y_range = range(half_tile_size, map.shape[1], tile_size)
                else:
                    y_range = range(0, map.shape[1], tile_size)
                for tile_y in y_range:
                    s_avg = self._square_avg(map, tile_x, tile_y, tile_size)
                    perturb = (2*np.random.rand()-1)*rand_scale
                    self._map_set(map, tile_x, tile_y, s_avg + perturb)

            scale -= 1
            rand_scale /= 2.0

            map -= np.amin(map)
            map /= np.amax(map)
            
        return map

    def _threshold_map(self, threshold):
        self.pipe_map[self.pipe_map < threshold] = threshold
        self.pipe_map -= threshold
        self.pipe_map /= (1-threshold)
        return self.pipe_map


if __name__ == "__main__":

    deposit = DepositGenerator(32, 64, threshold=0.3)
    Z = deposit.pipe_map

    
    X = np.linspace(0, Z.shape[0], Z.shape[0], endpoint=False)
    Y = np.linspace(0, Z.shape[1], Z.shape[1], endpoint=False)
    (X, Y) = np.meshgrid(X,Y)


    fig = plt.figure()
    ax  = Axes3D(fig)
    ax.plot_surface(X, Y, Z.transpose(), rstride=1, cstride=1, cmap=cm.coolwarm) 
    plt.show()
    






