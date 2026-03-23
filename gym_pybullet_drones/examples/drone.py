import numpy as np
import time
import math
import random
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='drone.log',
        filemode='w'
    )

class Drone:
    GRID_SIZE = 10
    FALSE_NEGATIVE_RATE = 0.2
    DRONE_DISTANCE_PENALTY = 0.05
    DRONE_SPAWN_POINT = np.array([[0, 0, 2]])
    TOTAL_TREES = 20
    
    def __init__(self):

        self.env = CtrlAviary(drone_model=DroneModel.CF2X, 
                    num_drones=1, 
                    physics=Physics.PYB,
                    initial_xyzs= self.DRONE_SPAWN_POINT,
                    gui=True)

        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.obs, self.info = self.env.reset()

        # Divide ones by total grid size
        self.belief_map = np.ones((self.GRID_SIZE, self.GRID_SIZE)) / (self.GRID_SIZE * self.GRID_SIZE)
        self.hiker_id = self.place_trees_and_hiker()
    
    def place_trees_and_hiker(self):
        # Place hiker
        hiker_x, hiker_y = 5, 5
        hiker_id = p.loadURDF("sphere2.urdf", [hiker_x, hiker_y, 1], globalScaling=0.5)
        p.changeVisualShape(hiker_id, -1, rgbaColor=[1, 0, 0, 1])

        # Place trees
        current_trees = 0
        while current_trees < self.TOTAL_TREES:
            tree_x = random.randint(0, 10)
            tree_y = random.randint(0, 10)

            # Tree cannot be on top of a hiker or on the drone's starting position
            if (tree_x == hiker_x and tree_y == hiker_y):
                continue

            tree_id = p.loadURDF("cube_small.urdf", [tree_x, tree_y, 1], globalScaling=15)
            p.changeVisualShape(tree_id, -1, rgbaColor=[0, 1, 0, 1])
            current_trees += 1
        # Return for camera logic
        return hiker_id

    def move(self, x_prime, y_prime, z_prime):
        # First 3 of initial state are x, y, z
        initial_state = self.obs[0]
        x, y, z = initial_state[0], initial_state[1], initial_state[2]

        distance = math.sqrt(((x_prime) - x)**2 + ((y_prime) - y)**2 + ((z_prime) - z)**2)

        SCALING_FACTOR = 75
        end_range = math.floor(distance * SCALING_FACTOR)
        
        for i in range(1, end_range + 1):
            state = self.obs[0] 
            progress = i / end_range
            target_pos = np.array([x + (x_prime - x) * progress, 
                                y + (y_prime - y) * progress,
                                z + (z_prime - z) * progress,]) 

            # Pybullet drone motors
            action = np.zeros((1, 4))
            action[0, :], _, _ = self.ctrl.computeControlFromState(
                control_timestep=self.env.CTRL_TIMESTEP,
                state=state,
                target_pos=target_pos,
            )

            # Step
            self.obs, _, _, _, self.info = self.env.step(action)
            
            # Render
            self.env.render()
            time.sleep(self.env.CTRL_TIMESTEP)


    def hover_and_capture_picture(self, x, y, z, ticks = 350):
        mask = None

        for i in range(ticks):
            current_state = self.obs[0]
            target_pos = np.array([x, y, z])

            action = np.zeros((1, 4))
            action[0, :], _, _ = self.ctrl.computeControlFromState(
                control_timestep=self.env.CTRL_TIMESTEP,
                state=current_state , 
                target_pos=target_pos,
            )

            # Step 
            self.obs, _, _, _, self.info = self.env.step(action)
            
            # Render
            self.env.render()

            if (i == (ticks // 2)):
                mask = self.capture_picture(self.obs[0])
            
            time.sleep(self.env.CTRL_TIMESTEP)
        
        return mask
    
    def capture_picture(self, current_state):
        drone_x, drone_y, drone_z = current_state[0:3]
        
        camera_eye = [drone_x, drone_y, drone_z] 
        camera_target = [drone_x, drone_y, 0.0]
        up_vector = [0, 1, 0]
        
        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_eye, 
                                        cameraTargetPosition=camera_target, 
                                        cameraUpVector=up_vector)
        
        # Maybe shrink FOV it later?
        projection_matrix = p.computeProjectionMatrixFOV(fov=60.0,
                                                        aspect=1.0, 
                                                        nearVal=0.1, 
                                                        farVal=100.0)

        _, _, _, _, segmentation_mask = p.getCameraImage(
            width=64, 
            height=64, 
            viewMatrix=view_matrix, 
            projectionMatrix=projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        return segmentation_mask

    def update_belief_not_seen(self, grid_x, grid_y):
        prior_probability = self.belief_map[grid_x, grid_y]

        new_probability = prior_probability * self.FALSE_NEGATIVE_RATE

        self.belief_map[grid_x, grid_y] = new_probability

        # Normalize
        self.belief_map = self.belief_map / np.sum(self.belief_map)

    def get_next_position(self, x, y, z):
        current_best_utility = float('-inf')
        current_best_target = (0, 0)

        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if self.belief_map[i][j] > current_best_utility:
                    current_best_target = (i, j)

        return np.array([current_best_target[0], current_best_target[1], z])
    
    def run_simulation(self):
        CONFIDENCE_THRESHOLD = .90

        while True:
            if self.belief_map.max() >= CONFIDENCE_THRESHOLD:
                flaten_hiker_index = self.belief_map.argmax()
                hiker_index = np.unravel_index(flaten_hiker_index, self.belief_map.shape)
                return hiker_index
            
            current_position = self.obs[0]
            next_position = self.get_next_position(current_position)

            self.move(next_position)
            mask = self.hover_and_capture_picture(0, 0, 3.0)
            if self.hiker_id in mask:
                # call update_belief_seen
            else:
                # call update_belief_not_seen

 
        return None

