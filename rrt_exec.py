import time
import numpy as np

import mujoco
import mujoco.viewer

from robot import UR5

from rrt_old import rrt
import utilities


def pos_from_fk(q):
    dh_params = utilities.joint_to_dh(q)
    pose = utilities.forward_kinematics(dh_params)[-1][:3, 3]
    return pose

robot = UR5()
m = robot.model
d = robot.data

with open('targets.npy', 'rb') as f:
    path = np.load(f)


with mujoco.viewer.launch_passive(m, d) as viewer:
    time.sleep(5)
    start = time.time()
    i = 0
    while viewer.is_running():
        step_start = time.time()
        
        try:
            robot.inverse_kinematics(
                target= path[i],
                rotation=-np.eye(3,3).reshape(9,1),
                )
        except:
            pass

        mujoco.mj_step(m, d)
        
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        if time.time() - start > i*1.5+1:
            i +=1

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)