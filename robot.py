import mujoco
import numpy as np

class UR5:
    def __init__(self,
                 integration_dt: float = .10,
                 damping: float = 1e-4,
                 dt: float = 0.002,
                 max_angvel = 0.0,
                 gravity_compensation: bool = True):
        
        self.integration_dt = integration_dt
        self.damping = damping
        self.dt = dt
        self.max_angvel = max_angvel
        self.gravity_compensation = gravity_compensation

        # Load the model and data.
        self.model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
        self.data = mujoco.MjData(self.model)

        # Override the simulation timestep.
        self.model.opt.timestep = dt

        # End-effector site we wish to control, in this case a site attached to the last
        # link (wrist_3_link) of the robot.
        self.site_id = self.model.site("attachment_site").id

        # Name of bodies we wish to apply gravity compensation to.
        self.body_names = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        body_ids = [self.model.body(name).id for name in self.body_names]
        if gravity_compensation:
            self.model.body_gravcomp[body_ids] = 1.0

        # Get the dof and actuator ids for the joints we wish to control.
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        # Note that actuator names are the same as joint names in this case.
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        self.key_id = self.model.key("home").id

        # Mocap body we will control with our mouse.
        self.mocap_id = self.model.body("target").mocapid[0]

        # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, self.model.nv))
        self.diag = damping * np.eye(6)
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

    def inverse_kinematics(self, target, rotation):
        self.error_pos[:] = target[:3] - self.data.site(self.site_id).xpos
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, rotation)
        # Orientation error.
        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        # Solve system of equations: J @ dq = error.
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)

        if self.max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > self.max_angvel:
                    dq *= self.max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = self.data.qpos.copy()
        
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
        
        # Set the control signal.
        np.clip(q[self.dof_ids], *self.model.jnt_range.T[:,self.dof_ids], out=q[self.dof_ids])
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
        print(self.data.ctrl[self.actuator_ids])
 
