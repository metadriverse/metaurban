import xml.etree.ElementTree as ET
import numpy as np
from skimage import measure
from metaurban.engine.logger import get_logger

import sys
sys.path.insert(0, './metaurban/orca_algo/build')
sys.path.append('/home/hollis/mnt/projects/MetaUrban-Private-for-Review/metaurban/orca_algo/build')

import bind
import metaurban.policy.orca_planner_utils as orca_planner_utils

logger = get_logger()
import time
import numpy as np


class OrcaPlanner:
    def __init__(self, uuid=None, ego=False):
        # TODO: simplify outside files
        import os
        os.makedirs('.cache', exist_ok=True)
        random_cache_file_id = np.random.uniform(0, 1)
        curr_time = time.time()
        if uuid is not None:
            self.template_xml_file = f'.cache/template_xml_file_{random_cache_file_id}_{curr_time}_{uuid}.xml'
        else:
            self.template_xml_file = f'.cache/template_xml_file_{random_cache_file_id}_{curr_time}.xml'
        self.uuid = uuid

        self.valid = False
        # self.num_agent = -1
        self.next_positions = []
        self.speed = []
        self.prev_start_positions, self.prev_goals = None, None
        self.earliest_stop_pos = []

    def generate_template_xml(self, mask):
        cellsize = 1
        agentdict = {"type": "orca-par-ecbs", "agent": []}

        mylist, h, w = orca_planner_utils.mask_to_2d_list(mask)
        contours = measure.find_contours(mylist, 0.5, positive_orientation='high')

        flipped_contours = []
        for contour in contours:
            contour = orca_planner_utils.find_tuning_point(contour, h)
            flipped_contours.append(contour)
        import os
        os.system(f'rm -rf ./.cache/{self.template_xml_file}')
        random_cache_file_id = np.random.uniform(0, 1)
        curr_time = time.time()
        if self.uuid is not None:
            self.template_xml_file = f'.cache/template_xml_file_{random_cache_file_id}_{curr_time}_{self.uuid}.xml'
        else:
            self.template_xml_file = f'.cache/template_xml_file_{random_cache_file_id}_{curr_time}.xml'

        orca_planner_utils.write_to_xml(mylist, w, h, cellsize, flipped_contours, agentdict, self.template_xml_file)

    def get_planning(self, start_positions, goals, num_agent, walkable_regions_mask=None):
        def get_speed(positions):
            pos1 = positions[:-1]
            pos2 = positions[1:]

            pos_delta = pos2 - pos1
            speed = np.linalg.norm(pos_delta, axis=2)
            speed = np.concatenate([np.zeros((1, len(start_positions))), speed], axis=0)
            return list(speed)

        self.set_agents(
            start_positions, goals, self.template_xml_file
        )  ## overwrite template_xml_file by new agent's starts & goal position
        result = bind.demo(self.template_xml_file, num_agent)

        # find shortest found path, return step number (index). At this index, store the current positions, --> ready to be next start points
        # min_total_step = min([item.total_step for item in result.values()])
        # min_total_step = min(100,min_total_step)
        nexts = []
        self.time_length_list = []
        for v in result.values():
            nextxr = np.array(v.xr)  #[:min_total_step]
            nextyr = np.array(v.yr)  #[:min_total_step]
            nextr = np.stack([nextxr, nextyr], axis=1)
            nexts.append(nextr)

            time_length = 0
            last_x, last_y = None, None
            flag = False
            for x, y in zip(nextxr, nextyr):
                if x == last_x and y == last_y:
                    self.time_length_list.append(time_length)
                    flag = True
                    break
                else:
                    last_x, last_y = x, y
                    time_length += 1
            if not flag:
                self.time_length_list.append(time_length)

        nexts = np.stack(nexts, axis=1)
        # print(min_total_step,nexts.shape[0])
        # assert min_total_step == nexts.shape[0]
        self.next_positions = list(nexts)
        self.speed = get_speed(nexts)

        # get stop_pos at index min_total_step
        self.earliest_stop_pos = self.next_positions[-1]

        # return next_positions, speed # --> next_positions.shape: [# steps, np.array(spawn_num,2)]

    def set_agents(self, start_positions, goals, template_xml_file):
        ### TODO, overwrite agent, instead of append
        ## overwrite agents' start and goal position in xml file
        tree = ET.parse(template_xml_file)
        root = tree.getroot()
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
            agent.set('goal.xr', f'{goal[0]+0.5}')  # magic number
            agent.set('goal.yr', f'{goal[1]+0.5}')

            agents.append(agent)
        tree.write(template_xml_file)

    def get_next(self, return_speed=False):
        if not return_speed:
            if not self.has_next():
                return None
            return self.next_positions.pop(0)
        else:
            if not self.has_next():
                return None, None
            return self.next_positions.pop(0), self.speed.pop(0)

    def has_next(self):
        if len(self.next_positions) > 0:
            return True
        else:
            return False
