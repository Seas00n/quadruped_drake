from planners.simple import *
import os
import time
import numpy as np


class LegSineTrunkPlanner(LeafSystem):
    def __init__(self, trunk_geometry_frame_id):
        LeafSystem.__init__(self)
        self.frame_ids = trunk_geometry_frame_id
        self.DeclareAbstractOutputPort(
            "trunk_trajectory",
            lambda: AbstractValue.Make({}),
            self.SetTrunkOutputs)
        fpv = FramePoseVector()
        for frame in self.frame_ids:
            fpv.set_value(trunk_geometry_frame_id[frame],
                          RigidTransform())
        self.DeclareAbstractOutputPort(
            "trunk_geometry",
            lambda: AbstractValue.Make(fpv),
            self.SetGeometryOutputs)
        self.output_dict = {}

        self.SetSine(0)

        # self.u2_max = self.ComputeMaxControlInputs()

    # 重写SetTrunkOutputs方法通过字典设置状态轨迹
    def SetTrunkOutputs(self, context, output):
        self.output_dict = output.get_mutable_value()
        t = context.get_time()
        self.SetSine(t)

    def SetGeometryOutputs(self, context, output):
        fpv = output.get_mutable_value()
        fpv.clear()

        X_trunk = RigidTransform()
        X_trunk.set_rotation(RollPitchYaw(self.output_dict["rpy_body"]))
        X_trunk.set_translation(self.output_dict["p_body"])

        fpv.set_value(self.frame_ids["trunk"], X_trunk)

        for foot in ["lf", "rf", "lh", "rh"]:
            X_foot = RigidTransform()
            X_foot.set_translation(self.output_dict["p_%s" % foot])
            fpv.set_value(self.frame_ids[foot], X_foot)

    def SetSine(self, t):
        T = 5e-3*400
        w = 2 * np.pi / T
        amp = 0.0
        circle_p = [amp * np.sin(w * t), amp * np.sin(w * t), 0]
        circle_pd = [amp * w * np.cos(w * t), amp * w * np.cos(w * t), 0]
        circle_pdd = [-amp * w * w * np.sin(w * t), -amp * w * w * np.sin(w * t), 0]
        # Foot positions
        self.output_dict["p_lf"] = np.array(circle_p) + np.array([0.2, 0.11, 1.2])
        self.output_dict["p_rf"] = np.array(circle_p) + np.array([0.2, -0.11, 1.2])
        self.output_dict["p_lh"] = np.array(circle_p) + np.array([-0.2, 0.11, 1.2])
        self.output_dict["p_rh"] = np.array(circle_p) + np.array([-0.2, -0.11, 1.2])
        print("p_lf from planner{}".format(self.output_dict["p_lf"]))
        # Foot velocities
        self.output_dict["pd_lf"] = np.array(circle_pd)
        self.output_dict["pd_rf"] = np.array(circle_pd)
        self.output_dict["pd_lh"] = np.array(circle_pd)
        self.output_dict["pd_rh"] = np.array(circle_pd)
        print("pd_lf from planner{}".format(self.output_dict["pd_lf"]))
        # Foot accelerations
        self.output_dict["pdd_lf"] = np.array(circle_pdd)
        self.output_dict["pdd_rf"] = np.array(circle_pdd)
        self.output_dict["pdd_lh"] = np.array(circle_pdd)
        self.output_dict["pdd_rh"] = np.array(circle_pdd)

        self.output_dict["contact_states"] = [False, False, False, False]
        self.output_dict["f_cj"] = np.zeros((3, 4))
        self.output_dict["rpy_body"] = np.array([0, 0, 0])
        self.output_dict["p_body"] = np.array([0, 0, 1.5])

        self.output_dict["rpyd_body"] = np.array([0, 0, 0])
        self.output_dict["pd_body"] = np.array([0, 0, 0])

        self.output_dict["rpydd_body"] = np.array([0, 0, 0])
        self.output_dict["pdd_body"] = np.array([0, 0, 0])

        self.output_dict["u2_max"] = 0

    def ComputeMaxControlInputs(self):
        """
        Compute ||u_2||_inf, the maximum L2 norm of the control input, which
        we take to be foot and body accelerations.

        This can be used for approximate-simulation-based control strategies,
        which require Vdot <= gamma(||u_2||_inf)
        """
        u2_max = 0
        for data in self.towr_data:
            u2_i = np.linalg.norm(
                np.hstack([data.lf_pdd,
                           data.rf_pdd,
                           data.lh_pdd,
                           data.rh_pdd,
                           data.base_rpydd,
                           data.base_pdd])
            )
            if u2_i > u2_max:
                u2_max = u2_i
        return u2_max
