import time
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

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

# def move(x, y, z):
#     state = obs[0] if isinstance(obs, np.ndarray) else obs['0']

#     target_pos = np.array([x, y, z])
#     target_rpy = np.array([0, 0, 0]) 

#     current


for i in range(100000, 1, -1):
    # Depending on the version, obs might be a dictionary or a 2D array
    # We need the state vector for drone 0
    state = obs[0] if isinstance(obs, np.ndarray) else obs['0']

    # 4. Define the Target Position (x, y, z)
    
    target_pos = np.array([0, 1, 3])


    # # Circle
    # target_pos = np.array([np.cos(i/500), np.sin(i/500), 1.0])

    # Roll, pitch, yaw
    target_rpy = np.array([3 / (i * 10), 3 / (i * 10), 3 / (i * 10)]) 

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

# env.close()