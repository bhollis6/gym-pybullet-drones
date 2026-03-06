import time
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import math

# 1. Setup the Environment
env = CtrlAviary(drone_model=DroneModel.CF2X, 
                num_drones=1, 
                physics=Physics.PYB, 
                gui=True)

# 2. Setup the Controller (The "Pilot")
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

# 3. Initialize
obs = env.reset()
action = np.zeros((1, 4)) # Standard 4-motor input

# 3. Initialize correctly (unpack the tuple)
obs, info = env.reset()

def move(x_prime, y_prime, z_prime):
    obs, info = env.reset()
    # First 3 of initial state are x, y, z
    initial_state = obs[0]
    x, y, z = initial_state[0], initial_state[1], initial_state[2]

    distance = math.sqrt(((x_prime) - x)**2 + ((y_prime) - y)**2 + ((z_prime) - z)**2)

    print("DISTANCE", distance)

    SCALING_FACTOR = 33
    end_range = math.floor(distance * SCALING_FACTOR)
    for i in range(0, end_range):
        state = obs[0] 
        target_pos = np.array([x_prime * (i / end_range), y_prime * (i / end_range), z_prime * (i / end_range)]) 

        # 5. Calculate motor PWMs
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state, # Use the extracted state vector
            target_pos=target_pos,
        )

        # 6. Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 7. Standard PyBullet rendering logic
        env.render()
        time.sleep(env.CTRL_TIMESTEP)

    while True:
        state = obs[0] 
        target_pos = np.array([x_prime, y_prime, z_prime]) 

        # 5. Calculate motor PWMs
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state, # Use the extracted state vector
            target_pos=target_pos,
        )

        # 6. Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 7. Standard PyBullet rendering logic
        env.render()
        time.sleep(env.CTRL_TIMESTEP)
move(20,20,20)


env.close()