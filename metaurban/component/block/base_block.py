import logging
from metaurban.constants import Semantics, CameraTagStateKey
import math
import warnings
from abc import ABC
from typing import Dict

import numpy as np
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletConvexHullShape
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh
from panda3d.core import LPoint3f, Material
from panda3d.core import TextureStage
from panda3d.core import Vec3, LQuaternionf, RigidBodyCombiner, \
    SamplerState, NodePath, Texture
from panda3d.core import Vec4

from metaurban.base_class.base_object import BaseObject
from metaurban.component.road_network.node_road_network import NodeRoadNetwork
from metaurban.component.road_network.road import Road
from metaurban.constants import CollisionGroup
from metaurban.constants import MetaUrbanType, CamMask, PGLineType, PGLineColor, PGDrivableAreaProperty
from metaurban.constants import TerrainProperty
from metaurban.engine.asset_loader import AssetLoader
from metaurban.engine.core.physics_world import PhysicsWorld
from metaurban.engine.logger import get_logger
from metaurban.engine.physics_node import BaseRigidBodyNode, BaseGhostBodyNode
from metaurban.utils.coordinates_shift import panda_vector, panda_heading
from metaurban.utils.math import norm
from metaurban.utils.vertex import make_polygon_model

warnings.filterwarnings('ignore', 'invalid value encountered in intersection')

logger = get_logger()


class BaseBlock(BaseObject, PGDrivableAreaProperty, ABC):
    """
    Block is a driving area consisting of several roads
    Note: overriding the _sample() function to fill block_network/respawn_roads in subclass
    Call Block.construct_block() to add it to world
    """
    ID = "B"

    def __init__(
        self,
        block_index: int,
        global_network: NodeRoadNetwork,
        random_seed,
        ignore_intersection_checking=False,
    ):
        super(BaseBlock, self).__init__(str(block_index) + self.ID, random_seed, escape_random_seed_assertion=True)
        # block information
        assert self.ID is not None, "Each Block must has its unique ID When define Block"
        assert len(self.ID) == 1, "Block ID must be a character "

        self.block_index = block_index
        self.ignore_intersection_checking = ignore_intersection_checking

        # each block contains its own road network and a global network
        self._global_network = global_network
        self.block_network = self.block_network_type()

        # a bounding box used to improve efficiency x_min, x_max, y_min, y_max
        self._bounding_box = None

        # used to spawn npc
        self._respawn_roads = []
        self._block_objects = None

        # polygons representing crosswalk and sidewalk
        self.crosswalks = {}
        self.sidewalks_near_road_buffer = {}
        self.sidewalks_near_road = {}
        self.sidewalks = {}
        self.sidewalks_farfrom_road = {}
        self.sidewalks_farfrom_road_buffer = {}
        self.valid_region = {}

        if self.render:
            # sidewalk
            # near road buffer
            self.nearroad_buffer_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "asphalt", "diff_2k.png"))
            self.nearroad_buffer_texture.setWrapU(Texture.WM_repeat)
            self.nearroad_buffer_texture.setWrapV(Texture.WM_repeat)
            self.nearroad_buffer_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.nearroad_buffer_texture.setAnisotropicDegree(8)
            self.nearroad_buffer_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
            self.nearroad_buffer_normal.setWrapU(Texture.WM_repeat)
            self.nearroad_buffer_normal.setWrapV(Texture.WM_repeat)
            
            # near road
            self.nearroad_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "sci", "color.jpg"))
            self.nearroad_texture.setWrapU(Texture.WM_repeat)
            self.nearroad_texture.setWrapV(Texture.WM_repeat)
            self.nearroad_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.nearroad_texture.setAnisotropicDegree(8)
            self.nearroad_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
            self.nearroad_normal.setWrapU(Texture.WM_repeat)
            self.nearroad_normal.setWrapV(Texture.WM_repeat)
            
            # side
            self.side_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "color.png"))
            # self.side_texture.set_format(Texture.F_srgb)
            self.side_texture.setWrapU(Texture.WM_repeat)
            self.side_texture.setWrapV(Texture.WM_repeat)
            self.side_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.side_texture.setAnisotropicDegree(8)
            self.side_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
            # self.side_normal.set_format(Texture.F_srgb)
            self.side_normal.setWrapU(Texture.WM_repeat)
            self.side_normal.setWrapV(Texture.WM_repeat)
            
            # farfrom road buffer
            self.farfrom_buffer_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "sci", "color.jpg"))
            self.farfrom_buffer_texture.setWrapU(Texture.WM_repeat)
            self.farfrom_buffer_texture.setWrapV(Texture.WM_repeat)
            self.farfrom_buffer_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.farfrom_buffer_texture.setAnisotropicDegree(8)
            self.farfrom_buffer_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
            self.farfrom_buffer_normal.setWrapU(Texture.WM_repeat)
            self.farfrom_buffer_normal.setWrapV(Texture.WM_repeat)
            
            # farfrom road
            self.farfrom_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "asphalt", "diff_2k.png"))
            self.farfrom_texture.setWrapU(Texture.WM_repeat)
            self.farfrom_texture.setWrapV(Texture.WM_repeat)
            self.farfrom_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.farfrom_texture.setAnisotropicDegree(8)
            self.farfrom_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
            self.farfrom_normal.setWrapU(Texture.WM_repeat)
            self.farfrom_normal.setWrapV(Texture.WM_repeat)
            
            # valid boundary
            self.valid_region_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "sci", "color.jpg"))
            self.valid_region_texture.setWrapU(Texture.WM_repeat)
            self.valid_region_texture.setWrapV(Texture.WM_repeat)
            self.valid_region_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.valid_region_texture.setAnisotropicDegree(8)
            self.valid_region_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
            self.valid_region_normal.setWrapU(Texture.WM_repeat)
            self.valid_region_normal.setWrapV(Texture.WM_repeat)
            
            self.line_seg = make_polygon_model([(-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5)], 0)

    def _sample_topology(self) -> bool:
        """
        Sample a new topology to fill self.block_network
        """
        raise NotImplementedError

    def construct_block(
        self,
        root_render_np: NodePath,
        physics_world: PhysicsWorld,
        extra_config: Dict = None,
        no_same_node=True,
        attach_to_world=True
    ) -> bool:
        """
        Randomly Construct a block, if overlap return False
        """
        self.sample_parameters()

        if not isinstance(self.origin, NodePath):
            self.origin = NodePath(self.name)
        # else:
        #     print("Origin already exists: ", self.origin)

        self._block_objects = []
        if extra_config:
            assert set(extra_config.keys()).issubset(self.PARAMETER_SPACE.parameters), \
                "Make sure the parameters' name are as same as what defined in pg_space.py"
            raw_config = self.get_config(copy=True)
            raw_config.update(extra_config)
            self.update_config(raw_config)
        self._clear_topology()
        success = self._sample_topology()
        self._global_network.add(self.block_network, no_same_node)

        self._create_in_world()
        self.attach_to_world(root_render_np, physics_world)

        # self.draw_polygons_in_network_block(self.block_network)


        if not attach_to_world:
            self.detach_from_world(physics_world)

        return success

    def detach_from_world(self, physics_world: PhysicsWorld):
        """
        Detach the object from the scene graph but store it in the memory
        Args:
            physics_world: PhysicsWorld, engine.physics_world

        Returns: None

        """
        if self.is_attached():
            for obj in self._block_objects:
                obj.detach_from_world(physics_world)
        super(BaseBlock, self).detach_from_world(physics_world)

    def attach_to_world(self, parent_node_path: NodePath, physics_world: PhysicsWorld):
        """
        Attach the object to the scene graph
        Args:
            parent_node_path: which parent node to attach
            physics_world: PhysicsWorld, engine.physics_world

        Returns: None

        """
        if not self.is_attached():
            for obj in self._block_objects:
                obj.attach_to_world(self.engine.worldNP, physics_world)
        super(BaseBlock, self).attach_to_world(parent_node_path, physics_world)

    def destruct_block(self, physics_world: PhysicsWorld):
        self._clear_topology()
        self.detach_from_world(physics_world)

        self.origin.removeNode()
        self.origin = None

        self.dynamic_nodes.clear()
        self.static_nodes.clear()
        for obj in self._block_objects:
            obj.destroy()
        self._block_objects = None
        self.crosswalks = {}
        self.sidewalks_near_road_buffer = {}
        self.sidewalks_near_road = {}
        self.sidewalks = {}
        self.sidewalks_farfrom_road = {}
        self.sidewalks_farfrom_road_buffer = {}
        self.valid_region = {}

    def construct_from_config(self, config: Dict, root_render_np: NodePath, physics_world: PhysicsWorld):
        success = self.construct_block(root_render_np, physics_world, config)
        return success

    def get_respawn_roads(self):
        return self._respawn_roads

    def get_respawn_lanes(self):
        """
        return a 2-dim array [[]] to keep the lane index
        """
        ret = []
        for road in self._respawn_roads:
            lanes = road.get_lanes(self.block_network)
            ret.append(lanes)
        return ret

    def get_intermediate_spawn_lanes(self):
        """Return all lanes that can be used to generate spawn intermediate vehicles."""
        raise NotImplementedError()

    def _add_one_respawn_road(self, respawn_road: Road):
        assert isinstance(respawn_road, Road), "Spawn roads list only accept Road Type"
        self._respawn_roads.append(respawn_road)

    def _clear_topology(self):
        if len(self._global_network.graph.keys()) > 0:
            self._global_network -= self.block_network
        self.block_network.graph.clear()
        self.PART_IDX = 0
        self.ROAD_IDX = 0
        self._respawn_roads.clear()

    """------------------------------------- For Render and Physics Calculation ---------------------------------- """

    def _create_in_world(self, skip=False):
        """
        Create NodePath and Geom node to perform both collision detection and render

        Note: Override the create_in_world() function instead of this one, since this method severing as a wrapper to
        help improve efficiency
        """
        self.lane_line_node_path = NodePath(RigidBodyCombiner(self.name + "_lane_line"))
        
        self.sidewalk_node_path = NodePath(RigidBodyCombiner(self.name + "_sidewalk"))
        self.nearroad_node_path = NodePath(RigidBodyCombiner(self.name + "_nearroad"))
        self.farfromroad_node_path = NodePath(RigidBodyCombiner(self.name + "_farfromroad"))
        self.nearroad_buffer_node_path = NodePath(RigidBodyCombiner(self.name + "_nearroad_buffer"))
        self.farfromroad_buffer_node_path = NodePath(RigidBodyCombiner(self.name + "_farfromroad_buffer"))
        self.valid_region_node_path = NodePath(RigidBodyCombiner(self.name + "_valid_region"))
        
        self.crosswalk_node_path = NodePath(RigidBodyCombiner(self.name + "_crosswalk"))
        
        self.lane_node_path = NodePath(RigidBodyCombiner(self.name + "_lane"))

        if skip:  # for debug
            pass
        else:
            self.create_in_world()

        self.lane_line_node_path.flattenStrong()
        self.lane_line_node_path.node().collect()
        self.lane_line_node_path.hide(CamMask.AllOn)
        self.lane_line_node_path.show(CamMask.SemanticCam)

        self.sidewalk_node_path.flattenStrong()
        self.sidewalk_node_path.node().collect()
        self.nearroad_node_path.flattenStrong()
        self.nearroad_node_path.node().collect()
        self.farfromroad_node_path.flattenStrong()
        self.farfromroad_node_path.node().collect()
        self.nearroad_buffer_node_path.flattenStrong()
        self.nearroad_buffer_node_path.node().collect()
        self.farfromroad_buffer_node_path.flattenStrong()
        self.farfromroad_buffer_node_path.node().collect()
        self.valid_region_node_path.flattenStrong()
        self.valid_region_node_path.node().collect()
        if self.render:
            # np.setShaderInput("p3d_TextureBaseColor", self.side_texture)
            # np.setShaderInput("p3d_TextureNormal", self.side_normal)
            self.sidewalk_node_path.setTexture(self.side_texture)
            self.nearroad_node_path.setTexture(self.nearroad_texture)
            self.farfromroad_node_path.setTexture(self.farfrom_texture)
            self.nearroad_buffer_node_path.setTexture(self.nearroad_buffer_texture)
            self.farfromroad_buffer_node_path.setTexture(self.farfrom_buffer_texture)
            self.valid_region_node_path.setTexture(self.valid_region_texture)
                        
            ts = TextureStage("normal")
            ts.setMode(TextureStage.MNormal)
            self.sidewalk_node_path.setTexture(ts, self.side_normal)
            self.nearroad_node_path.setTexture(ts, self.nearroad_normal)
            self.farfromroad_node_path.setTexture(ts, self.farfrom_normal)
            self.nearroad_buffer_node_path.setTexture(ts, self.nearroad_buffer_normal)
            self.farfromroad_buffer_node_path.setTexture(ts, self.farfrom_buffer_normal)
            self.valid_region_node_path.setTexture(ts, self.valid_region_normal)
            
            material = Material()
            self.sidewalk_node_path.setMaterial(material, True)
            self.nearroad_node_path.setMaterial(material, True)
            self.farfromroad_node_path.setMaterial(material, True)
            self.nearroad_buffer_node_path.setMaterial(material, True)
            self.farfromroad_buffer_node_path.setMaterial(material, True)
            self.valid_region_node_path.setMaterial(material, True)
            
        self.crosswalk_node_path.flattenStrong()
        self.crosswalk_node_path.node().collect()
        self.crosswalk_node_path.hide(CamMask.AllOn)
        self.crosswalk_node_path.show(CamMask.SemanticCam)

        # only bodies reparent to this node
        self.lane_node_path.flattenStrong()
        self.lane_node_path.node().collect()

        self.origin.hide(CamMask.Shadow)

        self.sidewalk_node_path.reparentTo(self.origin)
        self.nearroad_node_path.reparentTo(self.origin)
        self.farfromroad_node_path.reparentTo(self.origin)
        self.farfromroad_buffer_node_path.reparentTo(self.origin)
        self.nearroad_buffer_node_path.reparentTo(self.origin)
        self.valid_region_node_path.reparentTo(self.origin)
        self.crosswalk_node_path.reparentTo(self.origin)
        self.lane_line_node_path.reparentTo(self.origin)
        self.lane_node_path.reparentTo(self.origin)

        # semantics
        self.sidewalk_node_path.setTag(CameraTagStateKey.Semantic, Semantics.SIDEWALK.label)
        self.nearroad_node_path.setTag(CameraTagStateKey.Semantic, Semantics.SIDEWALK.label)
        self.farfromroad_node_path.setTag(CameraTagStateKey.Semantic, Semantics.SIDEWALK.label)
        self.farfromroad_buffer_node_path.setTag(CameraTagStateKey.Semantic, Semantics.SIDEWALK.label)
        self.nearroad_buffer_node_path.setTag(CameraTagStateKey.Semantic, Semantics.SIDEWALK.label)
        self.valid_region_node_path.setTag(CameraTagStateKey.Semantic, Semantics.SIDEWALK.label)

        try:
            self._bounding_box = self.block_network.get_bounding_box()
        except:
            if len(self.block_network.graph) > 0:
                logging.warning("Can not find bounding box for it")
            self._bounding_box = None, None, None, None

        self._node_path_list.append(self.sidewalk_node_path)
        self._node_path_list.append(self.nearroad_node_path)
        self._node_path_list.append(self.farfromroad_node_path)
        self._node_path_list.append(self.farfromroad_buffer_node_path)
        self._node_path_list.append(self.nearroad_buffer_node_path)
        self._node_path_list.append(self.valid_region_node_path)
        self._node_path_list.append(self.crosswalk_node_path)
        self._node_path_list.append(self.lane_line_node_path)
        self._node_path_list.append(self.lane_node_path)

    def create_in_world(self):
        """
        Create lane in the panda3D world
        """
        raise NotImplementedError

    def add_body(self, physics_body):
        raise DeprecationWarning(
            "Different from common objects like vehicle/traffic sign, Block has several bodies!"
            "Therefore, you should create BulletBody and then add them to self.dynamics_nodes "
            "manually. See in construct() method"
        )

    def get_state(self) -> Dict:
        """
        The record of Block type is not same as other objects
        """
        return {}

    def set_state(self, state: Dict):
        """
        Block type can not set state currently
        """
        pass

    def _add_box_body(self, lane_start, lane_end, middle, parent_np: NodePath, line_type, line_color):
        raise DeprecationWarning("Useless, currently")
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        if PGLineType.prohibit(line_type):
            node_name = MetaUrbanType.LINE_SOLID_SINGLE_WHITE if line_color == PGLineColor.GREY else MetaUrbanType.LINE_SOLID_SINGLE_YELLOW
        else:
            node_name = MetaUrbanType.BROKEN_LINE
        body_node = BulletGhostNode(node_name)
        body_node.set_active(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        body_np = parent_np.attachNewNode(body_node)

        self._node_path_list.append(body_np)

        shape = BulletBoxShape(
            Vec3(length / 2, PGDrivableAreaProperty.LANE_LINE_WIDTH / 2, PGDrivableAreaProperty.LANE_LINE_GHOST_HEIGHT)
        )
        body_np.node().addShape(shape)
        mask = PGDrivableAreaProperty.CONTINUOUS_COLLISION_MASK if line_type != PGLineType.BROKEN else PGDrivableAreaProperty.BROKEN_COLLISION_MASK
        body_np.node().setIntoCollideMask(mask)
        self.static_nodes.append(body_np.node())

        body_np.setPos(panda_vector(middle, PGDrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
        direction_v = lane_end - lane_start
        # theta = -numpy.arctan2(direction_v[1], direction_v[0])
        theta = panda_heading(math.atan2(direction_v[1], direction_v[0]))

        body_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))

    @property
    def block_network_type(self):
        """
        There are two type of road network to describe the relation of all lanes, override this func to assign one when
        you are building your own block.
        return: roadnetwork
        """
        raise NotImplementedError

    def destroy(self):
        if self.block_network is not None:
            self.block_network.destroy()
            if self.block_network.graph is not None:
                self.block_network.graph.clear()
            self.block_network = None
        self.PART_IDX = 0
        self.ROAD_IDX = 0
        self._respawn_roads.clear()
        self.crosswalks = {}
        self.sidewalks_near_road_buffer = {}
        self.sidewalks_near_road = {}
        self.sidewalks = {}
        self.sidewalks_farfrom_road = {}
        self.sidewalks_farfrom_road_buffer = {}
        self.valid_region = {}
        self._global_network = None
        super(BaseBlock, self).destroy()

    # def __del__(self):
    # self.destroy()
    # logger.debug("{} is being deleted.".format(type(self)))

    @property
    def bounding_box(self):
        return self._bounding_box

    def _construct_sidewalk(self):
        """
        Construct the sidewalk with collision shape
        """
        if self.engine is None or (self.engine.global_config["show_sidewalk"]):
            for sidewalk_id, sidewalk in self.sidewalks.items():
                if len(sidewalk["polygon"]) == 0:
                    continue
                polygons = TerrainProperty.clip_polygon(sidewalk["polygon"])
                if polygons is None:
                    continue
                for polygon in polygons:
                    height = sidewalk.get("height", None)
                    if height is None:
                        height = PGDrivableAreaProperty.SIDEWALK_THICKNESS
                    z_pos = height / 2
                    np = make_polygon_model(polygon, height)
                    np.reparentTo(self.sidewalk_node_path)
                    np.setPos(0, 0, z_pos)

                    body_node = BaseRigidBodyNode(None, MetaUrbanType.BOUNDARY_SIDEWALK)
                    body_node.setKinematic(False)
                    body_node.setStatic(True)
                    body_np = self.sidewalk_node_path.attachNewNode(body_node)
                    body_np.setPos(0, 0, z_pos)
                    self._node_path_list.append(body_np)

                    geom = np.node().getGeom(0)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(geom)
                    shape = BulletTriangleMeshShape(mesh, dynamic=False)

                    body_node.addShape(shape)
                    self.dynamic_nodes.append(body_node)
                    body_node.setIntoCollideMask(CollisionGroup.Sidewalk)
                    self._node_path_list.append(np)

    def _construct_crosswalk(self):
        """
        Construct the crosswalk for semantic Cam
        """
        if self.engine is None or (self.engine.global_config["show_crosswalk"] and not self.engine.use_render_pipeline):
            for cross_id, crosswalk in self.crosswalks.items():
                if len(crosswalk["polygon"]) == 0:
                    continue
                polygons = TerrainProperty.clip_polygon(crosswalk["polygon"])
                if polygons is None:
                    continue
                for polygon in polygons:
                    np = make_polygon_model(polygon, 1.5)

                    body_node = BaseGhostBodyNode(cross_id, MetaUrbanType.CROSSWALK)
                    body_node.setKinematic(False)
                    body_node.setStatic(True)
                    body_np = self.crosswalk_node_path.attachNewNode(body_node)
                    # A trick allowing collision with sidewalk
                    body_np.setPos(0, 0, 1.5)
                    self._node_path_list.append(body_np)

                    geom = np.node().getGeom(0)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(geom)
                    shape = BulletTriangleMeshShape(mesh, dynamic=False)

                    body_node.addShape(shape)
                    self.static_nodes.append(body_node)
                    body_node.setIntoCollideMask(CollisionGroup.Crosswalk)
                    np.removeNode()
                    
    def _construct_nearroadsidewalk(self):
        """
        Construct the sidewalk with collision shape
        """
        if self.engine is None or (self.engine.global_config["show_sidewalk"]):
            for _, nearroad_sidewalk in self.sidewalks_near_road.items():
                if len(nearroad_sidewalk["polygon"]) == 0:
                    continue
                polygons = TerrainProperty.clip_polygon(nearroad_sidewalk["polygon"])
                if polygons is None:
                    continue
                for polygon in polygons:
                    height = nearroad_sidewalk.get("height", None)
                    if height is None:
                        height = PGDrivableAreaProperty.SIDEWALK_THICKNESS
                    z_pos = height / 2
                    np = make_polygon_model(polygon, height)
                    np.reparentTo(self.nearroad_node_path)
                    np.setPos(0, 0, z_pos)

                    body_node = BaseRigidBodyNode(None, MetaUrbanType.BOUNDARY_SIDEWALK)
                    body_node.setKinematic(False)
                    body_node.setStatic(True)
                    body_np = self.nearroad_node_path.attachNewNode(body_node)
                    body_np.setPos(0, 0, z_pos)
                    self._node_path_list.append(body_np)

                    geom = np.node().getGeom(0)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(geom)
                    shape = BulletTriangleMeshShape(mesh, dynamic=False)

                    body_node.addShape(shape)
                    self.dynamic_nodes.append(body_node)
                    body_node.setIntoCollideMask(CollisionGroup.Sidewalk)
                    self._node_path_list.append(np)
                    
    def _construct_farfromroadsidewalk(self):
        """
        Construct the sidewalk with collision shape
        """
        if self.engine is None or (self.engine.global_config["show_sidewalk"]):
            for _, farfromroad_sidewalk in self.sidewalks_farfrom_road.items():
                if len(farfromroad_sidewalk["polygon"]) == 0:
                    continue
                polygons = TerrainProperty.clip_polygon(farfromroad_sidewalk["polygon"])
                if polygons is None:
                    continue
                for polygon in polygons:
                    height = farfromroad_sidewalk.get("height", None)
                    if height is None:
                        height = PGDrivableAreaProperty.SIDEWALK_THICKNESS
                    z_pos = height / 2
                    np = make_polygon_model(polygon, height)
                    np.reparentTo(self.farfromroad_node_path)
                    np.setPos(0, 0, z_pos)

                    body_node = BaseRigidBodyNode(None, MetaUrbanType.BOUNDARY_SIDEWALK)
                    body_node.setKinematic(False)
                    body_node.setStatic(True)
                    body_np = self.farfromroad_node_path.attachNewNode(body_node)
                    body_np.setPos(0, 0, z_pos)
                    self._node_path_list.append(body_np)

                    geom = np.node().getGeom(0)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(geom)
                    shape = BulletTriangleMeshShape(mesh, dynamic=False)

                    body_node.addShape(shape)
                    self.dynamic_nodes.append(body_node)
                    body_node.setIntoCollideMask(CollisionGroup.Sidewalk)
                    self._node_path_list.append(np)
                    
    def _construct_nearroadsidewalk_buffer(self):
        """
        Construct the sidewalk with collision shape
        """
        if self.engine is None or (self.engine.global_config["show_sidewalk"]):
            for _, nearroad_sidewalk in self.sidewalks_near_road_buffer.items():
                if len(nearroad_sidewalk["polygon"]) == 0:
                    continue
                polygons = TerrainProperty.clip_polygon(nearroad_sidewalk["polygon"])
                if polygons is None:
                    continue
                for polygon in polygons:
                    height = nearroad_sidewalk.get("height", None)
                    if height is None:
                        height = PGDrivableAreaProperty.SIDEWALK_THICKNESS
                    z_pos = height / 2
                    np = make_polygon_model(polygon, height)
                    np.reparentTo(self.nearroad_buffer_node_path)
                    np.setPos(0, 0, z_pos)

                    body_node = BaseRigidBodyNode(None, MetaUrbanType.BOUNDARY_SIDEWALK)
                    body_node.setKinematic(False)
                    body_node.setStatic(True)
                    body_np = self.nearroad_buffer_node_path.attachNewNode(body_node)
                    body_np.setPos(0, 0, z_pos)
                    self._node_path_list.append(body_np)

                    geom = np.node().getGeom(0)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(geom)
                    shape = BulletTriangleMeshShape(mesh, dynamic=False)

                    body_node.addShape(shape)
                    self.dynamic_nodes.append(body_node)
                    body_node.setIntoCollideMask(CollisionGroup.Sidewalk)
                    self._node_path_list.append(np)
                    
    def _construct_farfromroadsidewalk_buffer(self):
        """
        Construct the sidewalk with collision shape
        """
        if self.engine is None or (self.engine.global_config["show_sidewalk"]):
            for _, farfromroad_sidewalk in self.sidewalks_farfrom_road_buffer.items():
                if len(farfromroad_sidewalk["polygon"]) == 0:
                    continue
                polygons = TerrainProperty.clip_polygon(farfromroad_sidewalk["polygon"])
                if polygons is None:
                    continue
                for polygon in polygons:
                    height = farfromroad_sidewalk.get("height", None)
                    if height is None:
                        height = PGDrivableAreaProperty.SIDEWALK_THICKNESS
                    z_pos = height / 2
                    np = make_polygon_model(polygon, height)
                    np.reparentTo(self.farfromroad_buffer_node_path)
                    np.setPos(0, 0, z_pos)

                    body_node = BaseRigidBodyNode(None, MetaUrbanType.BOUNDARY_SIDEWALK)
                    body_node.setKinematic(False)
                    body_node.setStatic(True)
                    body_np = self.farfromroad_buffer_node_path.attachNewNode(body_node)
                    body_np.setPos(0, 0, z_pos)
                    self._node_path_list.append(body_np)

                    geom = np.node().getGeom(0)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(geom)
                    shape = BulletTriangleMeshShape(mesh, dynamic=False)

                    body_node.addShape(shape)
                    self.dynamic_nodes.append(body_node)
                    body_node.setIntoCollideMask(CollisionGroup.Sidewalk)
                    self._node_path_list.append(np)
                    
    def _construct_valid_region(self):
        """
        Construct the sidewalk with collision shape
        """
        if self.engine is None or (self.engine.global_config["show_sidewalk"]):
            for _, valid_region in self.valid_region.items():
                if len(valid_region["polygon"]) == 0:
                    continue
                polygons = TerrainProperty.clip_polygon(valid_region["polygon"])
                if polygons is None:
                    continue
                for polygon in polygons:
                    height = valid_region.get("height", None)
                    if height is None:
                        height = PGDrivableAreaProperty.SIDEWALK_THICKNESS
                    z_pos = height / 2
                    np = make_polygon_model(polygon, height)
                    np.reparentTo(self.valid_region_node_path)
                    np.setPos(0, 0, z_pos)

                    body_node = BaseRigidBodyNode(None, MetaUrbanType.BOUNDARY_SIDEWALK)
                    body_node.setKinematic(False)
                    body_node.setStatic(True)
                    body_np = self.valid_region_node_path.attachNewNode(body_node)
                    body_np.setPos(0, 0, z_pos)
                    self._node_path_list.append(body_np)

                    geom = np.node().getGeom(0)
                    mesh = BulletTriangleMesh()
                    mesh.addGeom(geom)
                    shape = BulletTriangleMeshShape(mesh, dynamic=False)

                    body_node.addShape(shape)
                    self.dynamic_nodes.append(body_node)
                    body_node.setIntoCollideMask(CollisionGroup.Sidewalk)
                    self._node_path_list.append(np)

    def _construct_lane(self, lane, lane_index):
        """
        Construct a physics body for the lane localization
        """
        if lane_index is not None:
            lane.index = lane_index
        # build physics contact
        if not lane.need_lane_localization:
            return
        assert lane.polygon is not None, "Polygon is required for building lane"
        if self.engine and self.engine.global_config["cull_lanes_outside_map"]:
            polygons = TerrainProperty.clip_polygon(lane.polygon)
            if not polygons:
                return
        else:
            polygons = [lane.polygon]
        for polygon in polygons:
            # It might be Lane surface intersection
            n = BaseRigidBodyNode(lane_index, lane.metaurban_type)
            segment_np = NodePath(n)

            self._node_path_list.append(segment_np)
            self._node_path_list.append(n)

            segment_node = segment_np.node()
            segment_node.set_active(False)
            segment_node.setKinematic(False)
            segment_node.setStatic(True)
            shape = BulletConvexHullShape()
            for point in polygon:
                # Panda coordinate is different from metaurban coordinate
                point_up = LPoint3f(*point, 0.0)
                shape.addPoint(LPoint3f(*point_up))
                point_down = LPoint3f(*point, -0.1)
                shape.addPoint(LPoint3f(*point_down))
            segment_node.addShape(shape)
            self.static_nodes.append(segment_node)
            segment_np.reparentTo(self.lane_node_path)

    def _construct_lane_line_segment(self, start_point, end_point, line_color: Vec4, line_type: PGLineType):
        node_path_list = []

        if not isinstance(start_point, np.ndarray):
            start_point = np.array(start_point)
        if not isinstance(end_point, np.ndarray):
            end_point = np.array(end_point)

        length = norm(end_point[0] - start_point[0], end_point[1] - start_point[1])
        middle = (start_point + end_point) / 2

        if not TerrainProperty.point_in_map(middle):
            return node_path_list

        parent_np = self.lane_line_node_path
        if length <= 0:
            return []
        if PGLineType.prohibit(line_type):
            liane_type = MetaUrbanType.LINE_SOLID_SINGLE_WHITE if line_color == PGLineColor.GREY \
                else MetaUrbanType.LINE_SOLID_SINGLE_YELLOW
        else:
            liane_type = MetaUrbanType.LINE_BROKEN_SINGLE_WHITE if line_color == PGLineColor.GREY \
                else MetaUrbanType.LINE_BROKEN_SINGLE_YELLOW

        # add bullet body for it
        body_node = BaseGhostBodyNode(None, liane_type)
        body_node.setActive(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        body_np = parent_np.attachNewNode(body_node)
        node_path_list.append(body_np)
        node_path_list.append(body_node)

        # its scale will change by setScale
        body_height = PGDrivableAreaProperty.LANE_LINE_GHOST_HEIGHT
        shape = BulletBoxShape(Vec3(length / 2, PGDrivableAreaProperty.LANE_LINE_WIDTH / 4, body_height))
        body_np.node().addShape(shape)
        mask = PGDrivableAreaProperty.CONTINUOUS_COLLISION_MASK if line_type != PGLineType.BROKEN \
            else PGDrivableAreaProperty.BROKEN_COLLISION_MASK
        body_np.node().setIntoCollideMask(mask)
        self.static_nodes.append(body_np.node())

        # position and heading
        body_np.setPos(panda_vector(middle, PGDrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
        direction_v = end_point - start_point
        # theta = -numpy.arctan2(direction_v[1], direction_v[0])
        theta = panda_heading(math.atan2(direction_v[1], direction_v[0]))
        body_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))

        return node_path_list


    def draw_polygons_in_network_block(self, network_block):
        """
        Visualize the polygons  with matplot lib
        Args:
            polygon: a list of 2D points

        Returns: None

        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(100, 100))

        for x in network_block.graph.keys():
            for y in network_block.graph[x].keys():
                for i in range(3):

                    polygon = network_block.graph[x][y][i].polygon

                    # Create the rectangle
                    rectangle_points = np.array(polygon)

                    # Plot the rectangle
                    plt.plot(*zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)), marker='o', label='['+x+']'+'['+y+']'+'['+str(i)+']', c=np.random.rand(1, 3))

                    # Fill the rectangle with light opacity
                    plt.fill(
                        *zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)), alpha=0.3, c=np.random.rand(1, 3)
                    )

        # Set equal scaling and labels
        plt.axis('equal')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Visualization of the Rectangle and Input Points')
        plt.legend()
        plt.grid(True)

        plt.show()

        input("Press Enter to continue...")