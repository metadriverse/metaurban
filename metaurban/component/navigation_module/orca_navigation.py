from collections import deque
from metaurban.constants import CamMask
import cv2, math
import numpy as np
from panda3d.core import NodePath, Material
from metaurban.engine.logger import get_logger
from metaurban.component.navigation_module.base_navigation import BaseNavigation
from metaurban.engine.asset_loader import AssetLoader
from metaurban.utils.coordinates_shift import panda_vector
from metaurban.utils.math import norm, clip
from metaurban.utils.math import panda_vector
from metaurban.utils.math import wrap_to_pi
from metaurban.policy.orca_planner import OrcaPlanner
import metaurban.policy.orca_planner_utils as orca_planner_utils
import torch
import numpy as np
import os.path as osp
from metaurban.engine.engine_utils import get_global_config
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.engine.logger import get_logger
from stable_baselines3 import PPO
from metaurban.policy.get_planning import get_planning
from metaurban.utils.math import panda_vector
from metaurban.engine.logger import get_logger
logger = get_logger()


def get_dest_heading(obj, dest_pos):
    position = obj.position

    dest = panda_vector(dest_pos[0], dest_pos[1])
    vec_to_2d = dest - position
    # dist_to = vec_to_2d.length()
    #### 
    
    heading = Vec2(*obj.heading).signedAngleDeg(vec_to_2d)
    #####
    return heading

class ORCATrajectoryNavigation(BaseNavigation):
    """
    This module enabling follow a given reference trajectory given a map
    """
    DISCRETE_LEN = 2  # m
    CHECK_POINT_INFO_DIM = 2
    NUM_WAY_POINT = 10
    NAVI_POINT_DIST = 30  # m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT

    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        self.mask_delta = 2
        self.sidewalks = {}
        self.crosswalks = {}
        self.walkable_regions_mask = None
        if show_dest_mark or show_line_to_dest:
            get_logger().warning("show_dest_mark and show_line_to_dest are not supported in ORCATrajectoryNavigation")
        super(ORCATrajectoryNavigation, self).__init__(
            show_navi_mark=False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        if self.origin is not None:
            self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)

        self._route_completion = 0
        self.checkpoints = None  # All check points
        
        seed = self.engine.global_random_seed
        import os, random
        import numpy as np
        import torch
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.walkable_regions_mask = self._get_walkable_regions(self.map)
        self.start_points, self.end_points = self.random_start_and_end_points(self.walkable_regions_mask[:, :, 0], 1)

        # for compatibility
        self.next_ref_lanes = None

        # override the show navi mark function here
        self._navi_point_model = None
        self._ckpt_vis_models = None
        if show_navi_mark and self._show_navi_info:
            self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_WAY_POINT)]
            for model in self._ckpt_vis_models:
                if self._navi_point_model is None:
                    self._navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                    self._navi_point_model.setScale(0.5)
                    # if self.engine.use_render_pipeline:
                    material = Material()
                    material.setBaseColor((19 / 255, 212 / 255, 237 / 255, 1))
                    material.setShininess(16)
                    material.setEmission((0.2, 0.2, 0.2, 0.2))
                    self._navi_point_model.setMaterial(material, True)
                self._navi_point_model.instanceTo(model)
                model.reparentTo(self.origin)

        # should be updated every step after calling update_localization
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)
    
    # TODO
    # No other objects
    # A*/other algos
    # * sidewalk centric coordinates
    # # 1. ego-orca
    # # 2. multi-agent orca
    
    def get_box_pts_from_center_heading(self, length, width, xc, yc, heading):
        import numpy as np
        def _rotate_pt(x, y, a):
            return np.cos(a)*x - np.sin(a)*y, np.sin(a)*x + np.cos(a)*y

        l, w = length / 2.0 , width / 2.0

        ## box
        x1, y1 = l, w
        x2, y2 = l, -w
        x3, y3 = -l, -w
        x4, y4 = -l, w

        ## rotation
        a = heading
        x1_, y1_ = _rotate_pt(x1, y1, a)
        x2_, y2_ = _rotate_pt(x2, y2, a)
        x3_, y3_ = _rotate_pt(x3, y3, a)
        x4_, y4_ = _rotate_pt(x4, y4, a)

        ## translation
        pt1 = [x1_ + xc, y1_ + yc]
        pt2 = [x2_ + xc, y2_ + yc]
        pt3 = [x3_ + xc, y3_ + yc]
        pt4 = [x4_ + xc, y4_ + yc]

        return [pt1, pt2, pt3, pt4]
        

    def reset(self, vehicle):
        import numpy as np
        import cv2
        from shapely.geometry import Polygon
        if 'ref_traj_path' in self.engine.global_config and self.engine.global_config['ref_traj_path'] != '':
            import pickle
            position_list = pickle.load(open(self.engine.global_config['ref_traj_path'], 'rb'))
            assert self.engine.global_config['ref_traj_path'].split('_')[-1].split('.')[0].lower() == self.engine.global_config['map'].lower()
            self.position_list = [np.array(i).reshape(2, ) for i in position_list]
            # mask_before = self.walkable_regions_mask
            # cv2.imwrite('./tmp_before.png', self.walkable_regions_mask)
            
            self.init_position = self.position_list[0]
            
            super(ORCATrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
            self.set_route()
            
            self.ref_position_list = self.checkpoints
            heading_list = []
            for p in range(1, len(self.ref_position_list)):
                heading_list.append(
                    np.arctan2(self.ref_position_list[p][1] - self.ref_position_list[p - 1][1], 
                             self.ref_position_list[p][0] - self.ref_position_list[p - 1][0])
                )
            self.heading_list = heading_list
            self.ref_position_list = self.ref_position_list[:-1]
            assert len(self.ref_position_list) == len(self.heading_list)
            self.pop_path = [self.position_list[len(self.position_list) - 1 - i] for i in range(len(self.position_list))]
            
        else:
            seed = self.engine.global_random_seed
            import os, random
            import torch
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            self.walkable_regions_mask = self._get_walkable_regions(self.map)
            self.start_points, self.end_points = self.random_start_and_end_points(self.walkable_regions_mask[:, :, 0], 1)
            # self.walkable_regions_mask[:] = 255
            # print( self.start_points, self.end_points)
            # self.start_points = [(10, 0)]
            # self.end_points = [(20, 0)]
            time_length, points, speed, early_stop_points = get_planning(
                    [self.start_points],
                    
                    [self.walkable_regions_mask],
                    
                    [self.end_points],
                    
                    [len(self.start_points)],
                    
                    1
            )
            
            # case-1
            # start_point = self._to_block_coordinate(points[0][50][0]) 
            # end_point = self._to_block_coordinate(points[0][60][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([-direction[1], direction[0]])

            # # 生成四个垂直点
            # distance = 1.5  # 每个点离线的距离
            # points2 = [start_point + i * distance * perpendicular for i in [-1, 1]]
            
            # start_point = self._to_block_coordinate(points[0][120][0]) 
            # end_point = self._to_block_coordinate(points[0][125][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([-direction[1], direction[0]])
            # points2 += [start_point + i * distance * perpendicular for i in [-1, 1]]

            # # 打印结果
            # from metaurban.component.agents.pedestrian.pedestrian_type import SimplePedestrian
            # selected_humanoid_configs = []
            # for i, point in enumerate(points2, start=1):
            #     print(f"Point {i}: {point}")
            #     spawn_point = point#self._to_block_coordinate(point) 
            #     random_humanoid_config = {"spawn_position_heading": [spawn_point, np.arctan2(perpendicular[1], perpendicular[0])]}
            #     selected_humanoid_configs.append(random_humanoid_config)
            # for kk, v_config in enumerate(selected_humanoid_configs):
            #     humanoid_type = SimplePedestrian

            #     v_config.update(self.engine.global_config["traffic_vehicle_config"])
            #     random_v = self.engine.spawn_object(humanoid_type, vehicle_config=v_config)
            #     self.engine.asset_manager.spawned_objects[random_v.id] = random_v
            
            # case-2
            # start_point = self._to_block_coordinate(points[0][50][0]) 
            # end_point = self._to_block_coordinate(points[0][60][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([direction[0], direction[1]])

            # # 生成四个垂直点
            # distance = 1.5  # 每个点离线的距离
            # points2 = [start_point + i * distance * perpendicular for i in [-1, 1]]
            
            # start_point = self._to_block_coordinate(points[0][120][0]) 
            # end_point = self._to_block_coordinate(points[0][125][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([direction[0], direction[1]])
            # points2 += [start_point + i * distance * perpendicular for i in [-1, 1]]

            # # 打印结果
            # from metaurban.component.agents.pedestrian.pedestrian_type import SimplePedestrian
            # selected_humanoid_configs = []
            # for i, point in enumerate(points2, start=1):
            #     print(f"Point {i}: {point}")
            #     spawn_point = point#self._to_block_coordinate(point) 
            #     random_humanoid_config = {"spawn_position_heading": [spawn_point, np.arctan2(perpendicular[1], perpendicular[0])]}
            #     selected_humanoid_configs.append(random_humanoid_config)
            # for kk, v_config in enumerate(selected_humanoid_configs):
            #     humanoid_type = SimplePedestrian

            #     v_config.update(self.engine.global_config["traffic_vehicle_config"])
            #     random_v = self.engine.spawn_object(humanoid_type, vehicle_config=v_config)
            #     self.engine.asset_manager.spawned_objects[random_v.id] = random_v
            
            # case-3
            # start_point = self._to_block_coordinate(points[0][10][0]) 
            # end_point = self._to_block_coordinate(points[0][20][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([-direction[1], direction[0]])

            # # 生成四个垂直点
            # distance = 1.5  # 每个点离线的距离
            # points2 = [start_point + i * distance * perpendicular for i in [-1, -0.8]]
            
            # start_point = self._to_block_coordinate(points[0][30][0]) 
            # end_point = self._to_block_coordinate(points[0][40][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([-direction[1], direction[0]])
            # points2 += [start_point + i * distance * perpendicular for i in [-1, -0.8]]

            # # 打印结果
            # from metaurban.component.agents.pedestrian.pedestrian_type import SimplePedestrian
            # selected_humanoid_configs = []
            # for i, point in enumerate(points2, start=1):
            #     print(f"Point {i}: {point}")
            #     spawn_point = point#self._to_block_coordinate(point) 
            #     random_humanoid_config = {"spawn_position_heading": [spawn_point, np.arctan2(perpendicular[1], perpendicular[0])]}
            #     selected_humanoid_configs.append(random_humanoid_config)
            # for kk, v_config in enumerate(selected_humanoid_configs):
            #     humanoid_type = SimplePedestrian

            #     v_config.update(self.engine.global_config["traffic_vehicle_config"])
            #     random_v = self.engine.spawn_object(humanoid_type, vehicle_config=v_config)
            #     self.engine.asset_manager.spawned_objects[random_v.id] = random_v
            
            # case-4
            # start_point = self._to_block_coordinate(points[0][50][0]) 
            # end_point = self._to_block_coordinate(points[0][60][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([-direction[1], direction[0]])

            # # 生成四个垂直点
            # distance = 1.5  # 每个点离线的距离
            # points2 = [start_point + i * distance * perpendicular for i in [-1, -0.8]]
            
            # start_point = self._to_block_coordinate(points[0][70][0]) 
            # end_point = self._to_block_coordinate(points[0][80][0]) 

            # # 计算方向向量
            # direction = end_point - start_point
            # direction = direction / np.linalg.norm(direction)  # 归一化

            # # 计算垂直向量
            # perpendicular = np.array([-direction[1], direction[0]])
            # points2 += [start_point + i * distance * perpendicular for i in [-1, -0.8]]

            # # 打印结果
            # from metaurban.component.agents.pedestrian.pedestrian_type import SimplePedestrian
            # selected_humanoid_configs = []
            # for i, point in enumerate(points2, start=1):
            #     print(f"Point {i}: {point}")
            #     spawn_point = point#self._to_block_coordinate(point) 
            #     random_humanoid_config = {"spawn_position_heading": [spawn_point, np.arctan2(perpendicular[1], perpendicular[0])]}
            #     selected_humanoid_configs.append(random_humanoid_config)
            # for kk, v_config in enumerate(selected_humanoid_configs):
            #     humanoid_type = SimplePedestrian

            #     v_config.update(self.engine.global_config["traffic_vehicle_config"])
            #     random_v = self.engine.spawn_object(humanoid_type, vehicle_config=v_config)
            #     self.engine.asset_manager.spawned_objects[random_v.id] = random_v
            
            
                
            positions = points[0]
            speeds = speed[0]
            self.position_list = []
            for p in positions:
                self.position_list.append(self._to_block_coordinate(p[0]))
            self.engine.ref_time_length = time_length[0][0]
            self.init_speed = speeds[0][0]
            self.init_position = self._to_block_coordinate(positions[0][0])
            map_mask = self.walkable_regions_mask[:, :, 0]
            if self.engine.global_config["show_ego_navigation"]:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                plt.imshow(np.flipud(self.walkable_regions_mask), origin='lower')   ######
                # plt.imshow(map_mask)
                ax.scatter([p[0] for p in self.end_points], [p[1] for p in self.end_points], marker='x')
                ax.scatter([p[0] for p in self.start_points], [p[1] for p in self.start_points], marker='o')
                plt.show()
            super(ORCATrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
            self.set_route()

    @property
    def reference_trajectory(self):
        
        if not hasattr(self, 'position_list'):
            self.position_list = []
        elif len(self.position_list) > 0:
            return self.get_idm_route(self.position_list)
                
        return self.get_idm_route(self.position_list)
    
    def _to_block_coordinate(self, point_in_mask: object) -> object:
        point_in_block = point_in_mask - self.mask_translate
        return point_in_block
    
    def random_start_and_end_points(self, map_mask, num):
        ### cv2.erode
        import os, random, torch, numpy as np
        seed = self.engine.global_random_seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        starts = self._random_points_new(map_mask, num)
        seed = self.engine.global_random_seed + 1
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        distance = 1.0
        import math
        iteration = 0
        # goals = self._random_points_new(map_mask, num, generated_position=starts[0])
        while distance <  5.0:
            seed = seed + 1
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            goals = self._random_points_new(map_mask, num, generated_position=starts[0])
            goal_pos = self._to_block_coordinate(goals[0])
            start_pos = self._to_block_coordinate(starts[0])
            distance = math.sqrt((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2)
            # print(iteration, distance, start_pos, goal_pos)
            iteration += 1
            if iteration > 100:
                break
        # import sys
        # sys.exit(0)
        #_random_points:  25: 0.1116s -0.133  | _random_points_new: 25: 0.008s

        #### visualization
        if self.engine.global_config["show_mid_block_map"]:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plt.imshow(np.flipud(map_mask), origin='lower')   ######
            # plt.imshow(map_mask)
            fixed_goal = ax.scatter([p[0] for p in goals], [p[1] for p in goals], marker='x')
            fixed_start = ax.scatter([p[0] for p in starts], [p[1] for p in starts], marker='o')
            plt.show()
        return starts, goals
    
    
    def _random_points_new(self, map_mask, num, min_dis=5, generated_position=None):
        import matplotlib.pyplot as plt
        from scipy.signal import convolve2d
        import random
        from skimage import measure
        h, _ = map_mask.shape
        import metaurban.policy.orca_planner_utils as orca_planner_utils
        mylist, h, w = orca_planner_utils.mask_to_2d_list(map_mask)
        contours = measure.find_contours(mylist, 0.5, positive_orientation='high')
        flipped_contours = []
        for contour in contours:
            contour = orca_planner_utils.find_tuning_point(contour, h)
            flipped_contours.append(contour)
        int_points = []
        for p in flipped_contours:
            for m in p:
                int_points.append((int(m[1]), int(m[0])))
        def find_walkable_area(map_mask):
            # kernel = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=np.uint8)
            kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
            conv_result= convolve2d(map_mask/255, kernel, mode='same')
            ct_pts = np.where(conv_result==8) #8, 24
            ct_pts = list(zip(ct_pts[1], ct_pts[0]))
            # print('Len Before:', len(ct_pts))
            ct_pts = [c for c in ct_pts if c not in int_points]
            # print('Len After:', len(ct_pts))
            # plt.imshow(map_mask, cmap='gray'); plt.scatter([pt[0] for pt in ct_pts], [pt[1] for pt in ct_pts], color='red')
            # plt.grid(True); plt.show()
            return ct_pts
        selected_pts = []
        walkable_pts = find_walkable_area(map_mask)
        if generated_position is not None:
            dis_to_start = np.linalg.norm(np.array([(x[0], h - 1 - x[1]) for x in walkable_pts]) - generated_position, axis=1)
            # print(dis_to_start[np.argsort(dis_to_start)[::-1]])
            walkable_pts = np.array(walkable_pts)[np.argsort(dis_to_start)[::-1]][:int(len(walkable_pts) / 10)].tolist()
        random.shuffle(walkable_pts)
        if len(walkable_pts) < num: raise ValueError(" Walkable points are less than spawn number! ")
        try_time = 0
        while len(selected_pts) < num:
            # print(try_time)
            if try_time > 10000: raise ValueError("Try too many time to get valid humanoid points!")
            cur_pt = random.choice(walkable_pts)
            if all(math.dist(cur_pt, selected_pt) >= min_dis for selected_pt in selected_pts): 
                selected_pts.append(cur_pt)
            try_time+=1
        selected_pts = [(x[0], h - 1 - x[1]) for x in selected_pts]
        return selected_pts
    
    def _get_walkable_regions(self, current_map):
        self.crosswalks = current_map.crosswalks
        self.sidewalks = current_map.sidewalks
        self.sidewalks_near_road = current_map.sidewalks_near_road
        self.sidewalks_farfrom_road = current_map.sidewalks_farfrom_road
        self.sidewalks_near_road_buffer = current_map.sidewalks_near_road_buffer
        self.sidewalks_farfrom_road_buffer = current_map.sidewalks_farfrom_road_buffer
        self.valid_region = current_map.valid_region

        polygons = []
        for sidewalk in self.sidewalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon = self.sidewalks[sidewalk]['polygon']
            polygons += polygon
        for crosswalk in self.crosswalks.keys():
            # if "CRS_I_" in crosswalk: continue
            polygon = self.crosswalks[crosswalk]['polygon']
            polygons += polygon
            
        for sidewalk in self.sidewalks_near_road_buffer.keys():
            polygon = self.sidewalks_near_road_buffer[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_near_road.keys():
            polygon = self.sidewalks_near_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road.keys():
            polygon = self.sidewalks_farfrom_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road_buffer.keys():
            polygon = self.sidewalks_farfrom_road_buffer[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.valid_region.keys():
            polygon = self.valid_region[sidewalk]['polygon']
            polygons += polygon

        polygon_array = np.array(polygons)
        min_x = np.min(polygon_array[:, 0])
        max_x = np.max(polygon_array[:, 0])
        min_y = np.min(polygon_array[:, 1])
        max_y = np.max(polygon_array[:, 1])

        rows = math.ceil(max_y - min_y) + 2*self.mask_delta
        columns = math.ceil(max_x - min_x) + 2*self.mask_delta

        self.mask_translate = np.array([-min_x+self.mask_delta, -min_y+self.mask_delta])
        if hasattr(self.engine, 'walkable_regions_mask'):
            return self.engine.walkable_regions_mask
        
        self.crosswalks = current_map.crosswalks
        self.sidewalks = current_map.sidewalks
        self.sidewalks_near_road = current_map.sidewalks_near_road
        self.sidewalks_farfrom_road = current_map.sidewalks_farfrom_road
        self.sidewalks_near_road_buffer = current_map.sidewalks_near_road_buffer
        self.sidewalks_farfrom_road_buffer = current_map.sidewalks_farfrom_road_buffer

        polygons = []
        for sidewalk in self.sidewalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon = self.sidewalks[sidewalk]['polygon']
            polygons += polygon
        for crosswalk in self.crosswalks.keys():
            # if "CRS_I_" in crosswalk: continue
            polygon = self.crosswalks[crosswalk]['polygon']
            polygons += polygon
            
        for sidewalk in self.sidewalks_near_road_buffer.keys():
            polygon = self.sidewalks_near_road_buffer[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_near_road.keys():
            polygon = self.sidewalks_near_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road.keys():
            polygon = self.sidewalks_farfrom_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road_buffer.keys():
            polygon = self.sidewalks_farfrom_road_buffer[sidewalk]['polygon']
            polygons += polygon

        polygon_array = np.array(polygons)
        min_x = np.min(polygon_array[:, 0])
        max_x = np.max(polygon_array[:, 0])
        min_y = np.min(polygon_array[:, 1])
        max_y = np.max(polygon_array[:, 1])

        rows = math.ceil(max_y - min_y) + 2*self.mask_delta
        columns = math.ceil(max_x - min_x) + 2*self.mask_delta

        self.mask_translate = np.array([-min_x+self.mask_delta, -min_y+self.mask_delta])
        walkable_regions_mask = np.zeros((rows, columns, 3), np.uint8)

        for sidewalk in self.sidewalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.sidewalks[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])

        for crosswalk in self.crosswalks.keys():
            # if "CRS_I_" in crosswalk: continue
            polygon_array = np.array(self.crosswalks[crosswalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        
        for sidewalk in self.sidewalks_near_road_buffer.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.sidewalks_near_road_buffer[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_near_road.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.sidewalks_near_road[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_farfrom_road.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.sidewalks_farfrom_road[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_farfrom_road_buffer.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.sidewalks_farfrom_road_buffer[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        walkable_regions_mask = cv2.flip(walkable_regions_mask, 0)   ### flip for orca   ###### 

        return walkable_regions_mask
    
    def get_map_mask(self):
        pass

    @property
    def current_ref_lanes(self):
        return None

    def set_route(self):
        self.checkpoints = self.discretize_reference_trajectory()
        num_way_point = min(len(self.checkpoints), self.NUM_WAY_POINT)
        waypoint_np = np.stack(self.checkpoints).reshape(-1, 2)
        if len(waypoint_np) > 1:
            moving_distance = np.linalg.norm(waypoint_np[1:] - waypoint_np[:-1], axis=-1).sum()
        else:
            moving_distance = 0.
        self.engine.agent_min_distance = moving_distance

        self._navi_info.fill(0.0)
        self.next_ref_lanes = None
        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], 1))

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DISCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.position(i * self.DISCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret
    
    def get_idm_route(self, traj_points, width=2):
        from metaurban.component.lane.point_lane import PointLane
        traj = PointLane(traj_points, width)
        return traj
    
    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        
        if self.reference_trajectory is None:
            return

        # Update ckpt index
        long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position)
        heading_theta_at_long = self.reference_trajectory.heading_theta_at(long)
        self.last_current_heading_theta_at_long.append(heading_theta_at_long)
        self.last_current_long.append(long)
        self.last_current_lat.append(lat)

        next_idx = max(int(long / self.DISCRETE_LEN) + 1, 0)
        next_idx = min(next_idx, len(self.checkpoints) - 1)
        end_idx = min(next_idx + self.NUM_WAY_POINT, len(self.checkpoints))
        ckpts = self.checkpoints[next_idx:end_idx]
        diff = self.NUM_WAY_POINT - len(ckpts)
        assert diff >= 0, "Number of Navigation points error!"
        if diff > 0:
            ckpts += [self.checkpoints[-1] for _ in range(diff)]

        # target_road_1 is the road segment the vehicle is driving on.
        self._navi_info.fill(0.0)
        for k, ckpt in enumerate(ckpts[1:]):
            start = k * self.CHECK_POINT_INFO_DIM
            end = (k + 1) * self.CHECK_POINT_INFO_DIM
            self._navi_info[start:end], lanes_heading = self._get_info_for_checkpoint(ckpt, ego_vehicle)
            if self._show_navi_info and self._ckpt_vis_models is not None:
                pos_of_goal = ckpt
                self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
                self._ckpt_vis_models[k].setH(self._goal_node_path.getH() + 3)

        self._navi_info[end] = clip((lat / self.engine.global_config["max_lateral_dist"] + 1) / 2, 0.0, 1.0)
        self._navi_info[end + 1] = clip(
            (wrap_to_pi(heading_theta_at_long - ego_vehicle.heading_theta) / np.pi + 1) / 2, 0.0, 1.0
        )

        # Use RC as the only criterion to determine arrival in Scenario env.
        self._route_completion = long / self.reference_trajectory.length

    @classmethod
    def _get_info_for_checkpoint(cls, checkpoint, ego_vehicle):
        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        dir_vec = checkpoint - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > cls.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * cls.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )  # project to ego vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(clip((ckpt_in_heading / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(clip((ckpt_in_rhs / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        return navi_information

    def destroy(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None
        super(ORCATrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None

    @property
    def route_completion(self):
        return self._route_completion

    @classmethod
    def get_navigation_info_dim(cls):
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM + 2

    @property
    def last_longitude(self):
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[1]
