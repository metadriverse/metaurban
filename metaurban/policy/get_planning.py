import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing
import bind
import math
import metaurban.policy.orca_planner_utils as orca_planner_utils
from skimage import measure
import time

def generate_template_xml( mask):
    cellsize = 1
    agentdict = {"type":"orca-par",  "agent": []}

    mylist, h, w = orca_planner_utils.mask_to_2d_list(mask)
    contours = measure.find_contours(mylist, 0.5, positive_orientation='high')

    flipped_contours = []
    for contour in contours:
        contour = orca_planner_utils.find_tuning_point(contour, h)
        flipped_contours.append(contour)
    root = orca_planner_utils.write_to_xml(mylist, w, h, cellsize, flipped_contours, agentdict)
    return root

def get_speed(start_positions, positions):
    pos1 = positions[:-1]
    pos2 = positions[1:]

    pos_delta = pos2 - pos1
    speed = np.linalg.norm(pos_delta, axis=2)
    speed = np.concatenate([np.zeros((1, len(start_positions))), speed], axis=0)
    return list(speed)
        
def set_agents(start_positions, goals, root):
    ### TODO, overwrite agent, instead of append
    ## overwrite agents' start and goal position in xml file
    agents = root.findall('./agents')[0]
    if agents.get("number") != "0":
        # print('need to overwrite agents')
        for child in agents.findall('agent'):
            agents.remove(child)
    agents.set("number", f"{len(start_positions)}")
    # num_agent = len(start_positions)

    for cnt, (pos, goal) in enumerate(zip(start_positions, goals)):
        agent = ET.Element("agent")
        agent.set('id', f'{cnt}')
        agent.set('size', f'{0.3}')
        agent.set('start.xr', f'{pos[0]}')
        agent.set('start.yr', f'{pos[1]}')
        agent.set('goal.xr', f'{goal[0]+0.5}') # magic number
        agent.set('goal.yr', f'{goal[1]+0.5}')

        agents.append(agent)
            
def run_planning(start_positions, goals, mask, num_agent, thread_id, results):
    root = generate_template_xml(mask)
    set_agents(start_positions, goals, root) 
    xml_string = ET.tostring(root, encoding='unicode')
    result = bind.demo(xml_string, num_agent)
    nexts = []
    time_length_list = []
    for v in result.values():
        nextxr = np.array(v.xr)#[:min_total_step] 
        nextyr = np.array(v.yr)#[:min_total_step]
        nextr = np.stack([nextxr, nextyr], axis=1)
        nexts.append(nextr)
        
        time_length = 0
        last_x, last_y = None, None
        flag = False
        for x, y in zip(nextxr, nextyr):
            if x == last_x and y == last_y:
                time_length_list.append(time_length)
                flag = True
                break
            else:
                last_x, last_y = x, y
                time_length += 1
        if not flag:
            time_length_list.append(time_length)
    nexts = np.stack(nexts, axis=1)
    speed = get_speed(start_positions, nexts)
    earliest_stop_pos = list(nexts)[-1]
    #print(f"Before assignment, results type: {type(results)}")
    #result = {key: convert_to_dict(value) for key, value in result.items()}
    #results[thread_id] = result
    results[thread_id] = (nexts, time_length_list, speed, earliest_stop_pos)
    #print(f"After assignment, results type: {type(results)}")


 
def get_planning(start_positions_list, masks, goals_list, num_agent_list, num_envs, roots=None):
    results = [None] * num_envs
    for i in range(num_envs):
        run_planning(start_positions_list[i], goals_list[i], masks[i], num_agent_list[i], i, results)
      
    nexts_list = []
    time_length_lists = []
    earliest_stop_pos_list = []
    speed_list = []
    for nexts, time_length_list, speed, earliest_stop_pos in results: 
        nexts_list.append(nexts)
        time_length_lists.append(time_length_list)
        speed_list.append(speed)
        earliest_stop_pos_list.append(earliest_stop_pos)

    # return next_positions, speed # --> next_positions.shape: [# steps, np.array(spawn_num,2)]
    return time_length_lists, nexts_list, speed_list, earliest_stop_pos_list
