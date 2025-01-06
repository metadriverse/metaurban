import numpy as np
import matplotlib.pyplot as plt 

polygons = np.load("polygons.npy",  allow_pickle=True)
print(polygons.size, polygons[0].size)
plt.figure()
for polygon in polygons:

    xcoords = polygon[:,0]
    ycoords = polygon[:,1]
    plt.plot(xcoords, ycoords, 'bo-')
    plt.fill(xcoords, ycoords, alpha=0.3)
plt.grid(True)
plt.show()