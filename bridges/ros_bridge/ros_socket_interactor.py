import argparse
import struct
import numpy as np
import zmq
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.constants import HELP_MESSAGE

class RosSocketInteractor(Node):
    def __init__(self):
        super().__init__('ros_socket_interactor')

        # Initialize ZMQ Context for Image Streaming
        self.zmq_context = zmq.Context().instance()
        self.zmq_context.setsockopt(zmq.IO_THREADS, 2)
        self.img_socket = self.zmq_context.socket(zmq.PUSH)
        self.img_socket.setsockopt(zmq.SNDBUF, 4194304)
        self.img_socket.bind("ipc:///tmp/rgb_camera")
        self.img_socket.set_hwm(5)

        # ROS2 Subscriber for Twist messages
        VEL_TOPIC = "/cmd_vel_mux/input/navi" # robot_config["vel_navi_topic"]
        self.subscription = self.create_subscription(
            Twist,
            VEL_TOPIC,
            self.twist_callback,
            10  # Queue size
        )
        self.subscription  # Prevent unused variable warning

        # Initialize Environment
        self.env = self.initialize_environment()
        self.latest_twist = [0.0, 0.0]  # Store latest velocity command

        # Timer for stepping the environment at a fixed rate
        self.timer = self.create_timer(0.1, self.step_env)  # NOTE: Run at 10Hz
        self.test = False
        self.get_logger().info(f"Subscribed to {VEL_TOPIC}, waiting for velocity commands...")

    def initialize_environment(self):
        """Initialize the MetaUrban environment with a predefined configuration."""
        config = dict(
            object_density=0.8,
            use_render=True,
            map='X',
            manual_control=True,
            crswalk_density=1,
            spawn_human_num=20,
            spawn_wheelchairman_num=1,
            spawn_edog_num=1,
            spawn_erobot_num=1,
            spawn_drobot_num=1,
            max_actor_num=1,
            drivable_area_extension=55,
            height_scale=1,
            spawn_deliveryrobot_num=2,
            show_mid_block_map=False,
            show_ego_navigation=False,
            debug=False,
            horizon=3000,
            norm_pixel=False,
            show_fps=False,
            on_continuous_line_done=False,
            out_of_route_done= False, # True,
            sensors={"camera": (RGBCamera, 480, 480)},
            image_observation=True,
            # image_on_cuda = True, # This is added by Rocky

            vehicle_config=dict(
                image_source="camera",
                show_navi_mark=True,
                show_lidar=False,
                show_line_to_navi_mark=True,
                show_dest_mark=True,
            ),
            show_sidewalk=True,
            show_crosswalk=True,
            random_spawn_lane_index=False,
            num_scenarios=100,
            traffic_density=0.2,
            accident_prob=0,
            window_size=(800, 800),
            relax_out_of_road_done=False, # True,
            max_lateral_dist=5.0,
        )

        env = SidewalkStaticMetaUrbanEnv(config)
        env.reset(seed=45)
        print(HELP_MESSAGE)
        env.agent.expert_takeover = False
        return env

    def twist_callback(self, msg):
        """Callback to handle received Twist messages."""
        self.latest_twist = [msg.linear.x, msg.angular.z]
        self.get_logger().info(f"Received velocity command - Linear: {msg.linear.x}, Angular: {msg.angular.z}")

    def step_env(self):
        """Step the environment using the latest received velocity command."""
        linear_x, angular_z = self.latest_twist
        action = np.array([angular_z, linear_x])
        action = np.clip(action, a_min=-1, a_max=1.)
        angular_z, linear_x = action.tolist()

        o = self.env.step([angular_z, linear_x])  # Apply action
        if self.test:
            image_data = np.zeros((512, 512, 3))  # fake data for testing
            image_data[::16, :, :] = 255
        else:
            image_data = o[0]['image'][..., -1]

        # Send camera data via ZeroMQ
        image_data = image_data.astype(np.uint8)
        dim_data = struct.pack('ii', image_data.shape[1], image_data.shape[0])
        image_data = dim_data + bytearray(image_data)

        try:
            self.img_socket.send(image_data, zmq.NOBLOCK)
        except zmq.error.Again:
            self.get_logger().warn("Error sending image data")

        # Check if simulation is done
        if o[2]:  # Done flag
            self.get_logger().info("Simulation ended, stopping node.")
            rclpy.shutdown()

    def run(self):
        rclpy.spin(self)

def main():
    rclpy.init()
    server = RosSocketInteractor()
    try:
        server.run()
    except KeyboardInterrupt:
        server.get_logger().info("Shutting down")
    finally:
        server.env.close()
        server.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main()
