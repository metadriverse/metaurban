def find_continuous_regions(binary_mask):
    def is_valid(x,y):
        return 0<=x<len(binary_mask[0]) and 0<=y<len(binary_mask)
    def bfs(x,y):
        queue = [(x,y)]
        visited.add((x,y))
        region = []
        while queue:
            currx, curry = queue.pop(0)
            region.append((currx, curry))
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                newx, newy = currx + dx, curry + dy
                if is_valid(newx,newy) and binary_mask[newy][newx] == 1 and (newx, newy) not in visited:
                    visited.add((newx, newy))
                    queue.append((newx,newy))
        return region
    visited = set()
    regions = []
    for y in range(len(binary_mask)):
        for x in range(len(binary_mask[0])):
            if binary_mask[y][x] == 1 and (x,y) not in visited:
                region = bfs(x,y)
                regions.append(region)
    return regions

def find_corner(regions):
    corners = []
    for region in regions:
        minx = min(vertex[0] for vertex in region)
        miny = min(vertex[1] for vertex in region)
        maxx = max(vertex[0] for vertex in region)
        maxy = max(vertex[1] for vertex in region)
        corners.append([(minx,miny),(maxx+1,miny),(maxx+1,maxy+1),(minx, maxy+1)])
    return corners

def find_obstacle_areas(binary_mask):
    obstacle_areas = []
    visited = set()

    def dfs(x,y):
        if ( 0<=x < len(binary_mask[0])
            and 0 <= y < len(binary_mask)
            and binary_mask[y][x] == 1
            and (x,y) not in visited):
            visited.add((x,y))
            return [(x,y)] + dfs(x-1,y) + dfs(x+1,y) + dfs(x,y+1) + dfs(x,y-1)
        return []
    
    for y in range(len(binary_mask)):
        for x in range(len(binary_mask[0])):
            if binary_mask[y][x] ==1 and (x,y) not in visited:
                area = dfs(x,y)
                obstacle_areas.append(area)
    return obstacle_areas


def find_tuning_points(obstacle_area):
    tuning_pts = []
    prev_dx, prev_dy = 0,0
    for i in range(1, len(obstacle_area)):
        dx = obstacle_area[i][0] - obstacle_area[i-1][0]
        dy = obstacle_area[i][1] - obstacle_area[i-1][1]
        if dx * prev_dy != dy * prev_dx:
            tuning_pts.append(obstacle_area[i-1])
        prev_dx, prev_dy = dx, dy
    return tuning_pts

binary_mask_2d_list = [[1,1,0],
                       [0,0,1]]
# binary_mask_2d_list = [
#     [1,1,0,0,0],
#     [0,1,1,0,0],
#     [1,0,0,0,1],
#     [1,0,1,0,1],
#     [1,1,1,1,1]
# ]
# binary_mask_2d_list = [
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ]

# binary_mask_2d_list = [
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
#         [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
#         [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
#         [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# ]
# (4,4),(12,4),(12,6),(4,6)
# (10,8),(12,8),(12,12),(10,12)
# (16,6),(17,6),(17,16),(16,16)
# (4,14),(5,14),(5,18),(4,18)

import numpy as np
binary_mask_2d_list = np.flipud(binary_mask_2d_list)

cregions = find_continuous_regions(binary_mask_2d_list)
vers = find_corner(cregions)
for i, ver in enumerate(vers):
    print(f' area {i+1}: {ver}')

# obstacle_areas = find_obstacle_areas(binary_mask_2d_list)
# print(len(obstacle_areas), obstacle_areas)
# obstacle_vertices = [find_tuning_points(area) for area in obstacle_areas]

# for i, vertices in enumerate(obstacle_vertices):
#     print(f' area {i+1}: {vertices}')


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.grid(color='gray', linestyle='-', linewidth=0.5, zorder=0, alpha=0.3)
img = ax.imshow(binary_mask_2d_list, cmap='binary',  
                extent=[0,len(binary_mask_2d_list[0]),0,len(binary_mask_2d_list)], origin='lower') #
plt.show()