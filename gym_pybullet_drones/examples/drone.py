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
    TOTAL_TREES = 800
    # 5, 6 challenging
    SEED = 1
    DRONE_DISTANCE_PENALTY = 0.0001
    DRONE_SPAWN_POINT = np.array([[0, 0, 15]])
    SLOWING_FACTOR = 50
    GRID_SIZE = 30
    DRONE_FOV = 45
    MAX_Z = 25.0
    MIN_Z = 5.0
    K_AGGRESSION = 80
    HIKER_MISS_LIMIT = 4

    def __init__(self):
        random.seed(self.SEED)
        self.env = CtrlAviary(drone_model=DroneModel.CF2X, 
                    num_drones=1, 
                    physics=Physics.PYB,
                    initial_xyzs= self.DRONE_SPAWN_POINT,
                    gui=True)
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.obs, self.info = self.env.reset()
        
        # Normalized belief map
        self.belief_map = np.ones((self.GRID_SIZE, self.GRID_SIZE)) / (self.GRID_SIZE * self.GRID_SIZE)
        self.hiker_cells = self.populate_hiker_cells()
        self.hiker_id = self.place_trees_and_hiker()    
    
    def calculate_false_negative_rate(self, x, y, z, i, j):
        distance = math.sqrt((y - i) ** 2 + (x - j) ** 2)
        radius = z * math.tan(math.radians(self.DRONE_FOV / 2))

        # Cap it at from 0.2 to 1
        distance_from_drone = max(0.2, min(distance / radius, 1))

        # z: [25, 15] - Med FNR
        # z: [14, 6] - Low FNR
        # z: [5]: - Extremlely low FNR

        if z >= 15:
            false_negative_rate = .20
        elif z >= 6:
            false_negative_rate = .05
        else:
            # Scanning, extremely low FNR
            false_negative_rate = .01
        
        return distance_from_drone * false_negative_rate

    def populate_hiker_cells(self):
        hiker_cells = set()
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                hiker_cells.add((i, j))
        return hiker_cells

    def place_trees_and_hiker(self):
        # Create hiker and tree id's
        tree_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.75], rgbaColor=[0, 1, 0, 1])
        hiker_x, hiker_y = random.randint(0, self.GRID_SIZE - 1), random.randint(0, self.GRID_SIZE - 1)
        hiker_id = p.loadURDF("sphere2.urdf", [hiker_x, hiker_y, 1], globalScaling=0.5)

        # Place hiker
        p.changeVisualShape(hiker_id, -1, rgbaColor=[1, 0, 0, 1])
        logging.info(f"Hiker spawning at: ({hiker_x}, {hiker_y})")

        # Place trees
        current_trees = 0
        while current_trees < self.TOTAL_TREES:
            tree_x = random.randint(0, self.GRID_SIZE - 1)
            tree_y = random.randint(0, self.GRID_SIZE - 1)
            tree_z = random.randint(1, 2)

            # Tree cannot be on top of a hiker
            if (tree_x == hiker_x and tree_y == hiker_y):
                continue

            p.createMultiBody(
                baseMass = 0,
                baseVisualShapeIndex=tree_id,
                baseCollisionShapeIndex=-1,
                basePosition=[tree_x, tree_y, tree_z]
            )
            current_trees += 1
        # Return for camera logic
        return hiker_id

    def get_distance(self, x, y, z, x_prime, y_prime, z_prime):
        return math.sqrt(((x_prime) - x)**2 + ((y_prime) - y)**2 + ((z_prime) - z)**2)

    def move(self, desired_stated):
        initial_state = self.obs[0]
        x, y, z = initial_state[0:3]
        x_prime, y_prime, z_prime = desired_stated[0:3]

        distance = self.get_distance(x, y, z, x_prime, y_prime, z_prime)
            
        # If drone is descending more than 3 units, descent slower.. (PyBullet drone logic will crash drone on a fast descent.)
        if (abs(z_prime - z) > 3):
            end_range = math.floor(distance * self.SLOWING_FACTOR * 1.5)
        # If drone is low altitude make it go a bit slower. (Prone to crashing at lower altitudes speeds)
        else:
            end_range = math.floor(distance * self.SLOWING_FACTOR)

        # Main movement loop
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
            p.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=0,
                cameraPitch=-85.9,
                cameraTargetPosition=[state[0], state[1], state[2]]
            )
            # Step
            self.obs, _, _, _, self.info = self.env.step(action)
            
            # Render
            self.env.render()
            time.sleep(self.env.CTRL_TIMESTEP)


    def hover_and_capture_picture(self, initial_state, ticks = 350):
        mask = None
        x_prime, y_prime, z_prime = initial_state[0:3]

        for i in range(ticks):
            current_state = self.obs[0]
            x, y, z = current_state[0:3]

            target_pos = np.array([x_prime, y_prime, z_prime])

            action = np.zeros((1, 4))
            action[0, :], _, _ = self.ctrl.computeControlFromState(
                control_timestep=self.env.CTRL_TIMESTEP,
                state=current_state , 
                target_pos=target_pos,
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=0,
                cameraPitch=-85.9,
                cameraTargetPosition=[x, y, z]
            )

            # Step 
            self.obs, _, _, _, self.info = self.env.step(action)
            
            # Render
            self.env.render()

            # Capture the picture near the final point in the hover
            if (i == (ticks // 2)):
                mask = self.capture_picture(self.obs[0])
            
            time.sleep(self.env.CTRL_TIMESTEP)
        
        return mask
    
    def capture_picture(self, current_state):
        x, y, z = current_state[0:3]
        
        camera_eye = [x, y, z] 
        camera_target = [x, y, 0.0]
        up_vector = [0, 1, 0]
        
        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_eye, 
                                        cameraTargetPosition=camera_target, 
                                        cameraUpVector=up_vector)
        
        projection_matrix = p.computeProjectionMatrixFOV(fov=self.DRONE_FOV,
                                                        aspect=1.0, 
                                                        nearVal=0.1, 
                                                        farVal=500.0)

        _, _, _, _, mask = p.getCameraImage(
            width=64, 
            height=64, 
            viewMatrix=view_matrix, 
            projectionMatrix=projection_matrix, 
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        captured_cells = self.get_captured_cells(x, y, z)

        return mask, captured_cells

    def get_captured_cells(self, x, y, z):
        radius = z * math.tan(math.radians(self.DRONE_FOV / 2))

        #want a radius of .5
        #DRONE_FOV = 2 * math.degrees(math.atan(radius / z))

        max_x = min(self.GRID_SIZE - 1, math.ceil(x + radius))
        min_x = max(0, math.floor(x - radius))
        max_y = min(self.GRID_SIZE - 1, math.ceil(y + radius))
        min_y = max(0, math.floor(y - radius))

        return (min_x, max_x, min_y, max_y)


    def update_belief_map(self, current_position, captured_cells, hiker_seen):
        min_x, max_x, min_y, max_y = captured_cells[0:4]
        def in_bounds(i, j):
            return min_x <= j <= max_x and min_y <= i <= max_y

        x, y, z = current_position[0:3]

        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                prior_probability = self.belief_map[i][j]
                if (i, j) in self.hiker_cells:
                    if hiker_seen:
                        if in_bounds(i, j):
                            self.belief_map[i][j] = prior_probability * (1 - self.calculate_false_negative_rate(x, y, z, i, j))
                            self.hiker_cells.add((i, j))
                        # Hiker seen, not in bounds
                        else:
                            self.hiker_cells.remove((i, j))
                            self.belief_map[i][j] = 0.0
                    
                    # Hiker not seen
                    else:
                        if in_bounds(i, j):
                            # Might have missed the hiker
                            self.belief_map[i][j] = prior_probability * self.calculate_false_negative_rate(x, y, z, i, j)
                        # Hiker not seen, not in_bounds
                        else:
                            # Normalization will make sure that these go up anyways
                            continue

        # Normalize and avoid a divide by zero.
        if np.sum(self.belief_map) > 0:
            self.belief_map = self.belief_map / (np.sum(self.belief_map))

    def get_next_position(self, current_state):
        x, y, z = current_state[0:3]

        best_utility = float('-inf')
        best_next_position = (0, 0)

        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if (i, j) in self.hiker_cells:     
                    current_utility = self.belief_map[i][j]

                    distance = math.sqrt((j - x)**2 + (i - y)**2)

                    current_utility = current_utility - (distance * self.DRONE_DISTANCE_PENALTY)

                    if current_utility > best_utility:
                        best_next_position = (j, i)
                        best_utility = current_utility
        # Had to use an shifted exp decay function. POMDP could not do this in real time.     
        best_next_z = self.MIN_Z + (self.MAX_Z - self.MIN_Z) * math.exp((-self.K_AGGRESSION) * (np.max(self.belief_map) - (1.0 / (self.GRID_SIZE * self.GRID_SIZE))))       
        
        logging.info(f"At {x} {y} {z}. Moving to {(best_next_position[0], best_next_position[1], best_next_z)}. Best belief map value {np.max(self.belief_map)}")

        return np.array([best_next_position[0], best_next_position[1], best_next_z])
        
    # Called to prevent thrashing
    def sweep(self):

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        # Find scanning grid

        for y, x in self.hiker_cells:
            min_x, max_x = min(min_x, x), max(max_x, x)

            min_y, max_y = min(min_y, y), max(max_y, y)
        
        # Expand for odd grid sizing
        if ((max_x - min_x) % 2) == 0:
            max_x += 1

        if ((max_y - min_y) % 2) == 0:
            max_y += 1

        logging.info(f"WINDOW (mix_x, max_x, min_y, max_y): {min_x}, {max_x}, {min_y}, {max_y}")
        
        x = min_x + 1
        y = min_y + 0.5

        self.move([x, y, self.MIN_Z])
        reverse = False

        # Want to scan the row / column with the maximum length
        # i.e 3x8 grid, want to scan the 3 long rows instead of the 8 short columns
        if ((max_x - min_x) >= (max_y - min_y)):
            # Row by row sweeping logic
            while y <= max_y:
                for i in range((max_x - min_x)):                    
                    mask, _ = self.hover_and_capture_picture([x, y, self.MIN_Z], 125)

                    logging.info(f"moving to {x, y, self.MIN_Z}")
                    if self.hiker_id in mask:
                        return True

                    if not reverse:
                            x += 1
                    else:
                            x -= 1
                    if i == ((max_x - min_x - 1)):
                        y += 2
                        if not reverse:
                            x -= 1
                        else:
                            x += 1
                        self.move([x, y, self.MIN_Z])
                    
                        reverse = not reverse
        else:
            # column by row sweeping logic
            while x <= max_x:
                for i in range((max_y - min_y)):
                    # may be able to omit
                    
                    mask, _ = self.hover_and_capture_picture([x, y, self.MIN_Z], 125)

                    logging.info(f"moving to {x, y, self.MIN_Z}")
                    if self.hiker_id in mask:
                        return True

                    if not reverse:
                            y += 1
                    else:
                            y -= 1
                    logging.info(f"i: {i}")
                    if i == ((max_y - min_y - 1)):
                        x += 2
                        if not reverse:
                            y -= 1
                        else:
                            y += 1
                        self.move([x, y, self.MIN_Z])
                    
                        reverse = not reverse

    # def scan(self, current_position):

    def run_simulation(self):
        hiker_seen = False
        hiker_ever_seen = False
        hiker_miss_count = 0
        mode = "SCANNING"

        start = time.perf_counter()
        while True:
            # HIKER_MISS_LIMIT tries to find the hiker before entering sweep mode
            if hiker_miss_count >= self.HIKER_MISS_LIMIT:
                mode = "SWEEPING"

            current_position = self.obs[0]
            logging.info(f"At {(current_position[0:3])}")

            if mode == "SCANNING":
                # Scan function
                next_position = self.get_next_position(current_position)
                self.move(next_position)
                logging.info(f"Moving to {(next_position[0:3])}")

                mask, captured_cells = self.hover_and_capture_picture(next_position)
                
                if self.hiker_id in mask:
                    if math.floor(current_position[2]) == 5:
                        break
                    hiker_seen = True
                    hiker_ever_seen = True
                    hiker_miss_count = 0
                    logging.info(f"FOUND in cells {captured_cells}")
                else:
                    hiker_seen = False
                    
                    if hiker_ever_seen:
                        hiker_miss_count += 1
                    logging.info(f"MISSING in cells {captured_cells}")
                
                self.update_belief_map(next_position, captured_cells, hiker_seen)
            else:
                if self.sweep():
                    break

            endtime = time.perf_counter()
            logging.info(f"Hiker Found in {endtime - start:.6f} seconds")

        # Victory hover
        self.hover_and_capture_picture(self.obs[0], 700)

        return True