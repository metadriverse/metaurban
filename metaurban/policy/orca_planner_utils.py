import xml.etree.ElementTree as ET
import sys
from metaurban.engine.logger import get_logger

logger = get_logger()

# sys.path.insert(0, 'metaurban/orca_algo/build')
# import bind
import numpy as np
import math



from PIL import Image, ImageOps
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import math


def write_to_xml(grid, width, height, cellsize, flipped_contours, agentdict, filename):
    root = ET.Element('root')

    agents = ET.SubElement(root,'agents')
    agents.set('number', str(len(agentdict['agent'])))
    agents.set('type', agentdict['type'])
    default_parameters = ET.SubElement(agents,'agents',
                                       size="0.3",
                                       movespeed="1",
                                       agentsmaxnum="10",
                                       timeboundary="5.4",
                                       sightradius="3.0",
                                       timeboundaryobst="33")

    for agent in agentdict['agent']:
        tmpagent = ET.SubElement(agents,"agent",
                            id=agent["id"],
                            size=agent["size"],
                            **{"start.xr":str(agent["start.xr"]),
                            "start.yr":str(agent["start.yr"]),
                            "goal.xr":str(agent["goal.xr"]),
                            "goal.yr":str(agent["goal.yr"])})
    obstacles = ET.SubElement(root,'obstacles')
    obstacles.set('number', str(len(flipped_contours)))
    for k, contour in enumerate(flipped_contours):
        obstacle = ET.SubElement(obstacles, 'obstacle')
        for pt in contour:
            xr, yr = pt
            vertex = ET.SubElement(obstacle,'vertex')
            vertex.set('xr', str(int(xr)))
            vertex.set('yr', str(int(yr)))

    map = ET.SubElement(root, "map")
    width_elem = ET.SubElement(map, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(map, "height")
    height_elem.text = str(height)
    cellsize_elem = ET.SubElement(map, "cellsize")
    cellsize_elem.text = str(cellsize)
    grid_elem = ET.SubElement(map, "grid")
    for row in grid:
        row_elem = ET.SubElement(grid_elem,"row")
        row_elem.text = " ".join(str(cell) for cell in row)

    algo = ET.SubElement(root, 'algorithm')
    searchtype = ET.SubElement(algo, 'searchtype')
    searchtype.text = 'thetastar'
    breakingties = ET.SubElement(algo, 'breakingties')
    breakingties.text = '0'
    allowsqueeze = ET.SubElement(algo, 'allowsqueeze')
    allowsqueeze.text = 'false'
    cutcorners = ET.SubElement(algo, 'cutcorners')
    cutcorners.text = 'false'
    hweight = ET.SubElement(algo, 'hweight')
    hweight.text = '1'
    timestep = ET.SubElement(algo, 'timestep')
    timestep.text = '0.1'
    delta = ET.SubElement(algo, 'delta')
    delta.text = '0.1'
    trigger = ET.SubElement(algo, 'trigger')
    trigger.text = 'speed-buffer'
    mapfnum = ET.SubElement(algo, 'mapfnum')
    mapfnum.text = '3'

    tree = ET.ElementTree(root)
    with open(filename, "wb") as f:
        f.write(prettify(root))

def prettify(elem):
    ## add indent in xml
    rough_str = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_str)
    return reparsed.toprettyxml(indent="   ", encoding="utf-8")

def mask_to_2d_list(mask):
    ## get <map> required by xml
    img = Image.fromarray(mask)
    img = img.convert('L')
    binary_array = img.point(lambda x:0 if x<128 else 1, mode='1')
    h = binary_array.size[1]
    w = binary_array.size[0]
    binary_list = []
    for y in range(h):
        row = []
        for x in range(w):
            row.append(1 - binary_array.getpixel((x,y))) ## revert 0 & 1
        binary_list.append(np.array(row))
    binary_list = np.array(binary_list)
    return binary_list, h, w

def find_tuning_point(contour, h):
    unique_pt = []
    filtered_contour = []
    ppp = len(contour)
    for i, (y,x) in enumerate(contour):
        if len(unique_pt) == 0 or (x != unique_pt[-1][0] and (h - 1 - y) != unique_pt[-1][1]):
            y = h - 1 - y    ######
            unique_pt.append([x,y])
    prev_len = len(unique_pt)
    unique_pt = remove_middle_points(unique_pt)
    return np.array(unique_pt)

def remove_middle_points(pts):
    ## eliminate middle points on the same straight line segment
    if len(pts)<3: return pts
    filtered = [pts[0]]
    prev_pt = pts[0]

    for i in range(1, len(pts)-1):
        next_pt = pts[i+1]
        if is_collinear(prev_pt, pts[i], next_pt): continue
        filtered.append(pts[i])

    filtered.append(pts[-1]) # add last point
    return filtered

def slope(p1, p2):
    ## calculate slope of point p1 and p2 -> (x,y)
    if (p2[0]- p1[0]) == 0: return np.inf
    elif (p2[1]- p1[1]) == 0: return 0
    else: return (p2[1]- p1[1]) / p2[0]- p1[0]

def is_collinear(p1, p2, p3):
    ## determine whether three points are col-linear
    return np.abs(slope(p1,p2) - slope(p2,p3)) < 0.1


def vis_orca(nexts, walkable_regions_mask, spawn_num, start_points, end_points):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    def update(frame):
        for i, points_list in enumerate(converted_nexts):
            if frame < len(points_list):
                x, y = points_list[frame]
                scatters[i].set_offsets((x,y))
            else:
                scatters[i].set_offsets([],[])
        return scatters

    ## convert nexts for visualization, now converted_nexts's shape is [spawn_num, np.array(1001 steps, 2)]
    converted_nexts = np.stack(nexts)
    converted_nexts = [converted_nexts[:,i,:] for i in range(converted_nexts.shape[1])]

    points_lists = []
    fig, ax = plt.subplots()
    plt.imshow(np.flipud(walkable_regions_mask), origin='lower')

    colors = []
    for i in range(spawn_num):
        colors.append(np.random.rand(1, 3))
    fixed_goal = ax.scatter([p[0] for p in end_points], [p[1] for p in end_points], marker='x', color=colors)
    fixed_start = ax.scatter([p[0] for p in start_points], [p[1] for p in start_points], marker='o', color=colors)
    scatters = [ax.scatter([],[], label=f'Agent {i+1}', color=colors[i]) for i in range(len(converted_nexts))]

    ani = animation.FuncAnimation(fig, update, frames=len(max(converted_nexts,key=len)), blit=True, interval=10)

    plt.show()

def update(frame, scatters, converted_nexts):
    for i, points_list in enumerate(converted_nexts):
        if frame < len(points_list):
            x, y = points_list[frame]
            scatters[i].set_offsets((x,y))
        else:
            scatters[i].set_offsets([],[])
    return scatters