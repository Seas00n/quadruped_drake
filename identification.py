from pydrake.all import (
    getDrakePath, FindResourceOrThrow,
    DiagramBuilder, SceneGraph,
    MultibodyPlant, Parser,
    RigidTransform, CoulombFriction, HalfSpace, GeometryFrame, GeometryInstance,
    Box, MakePhongIllustrationProperties, Sphere, LogVectorOutput, DrakeVisualizer,
    ConnectContactResultsToDrakeVisualizer,
    plot_system_graphviz, Simulator, RollPitchYaw,
)
from manipulation.scenarios import AddMultibodyTriad
from controllers import *
from planners import BasicTrunkPlanner, TowrTrunkPlanner, LegSineTrunkPlanner
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import io

############### Common Parameters ###################
show_trunk_model = True
use_lcm = False

planning_method = "leg_sine"  # "towr" or "basic" or "leg_sine"
control_method = "Identification"  # ID = Inverse Dynamics (standard QP),
# B = Basic (simple joint-space PD),
# MPTC = task-space passivity
# PC = passivity-constrained
# CLF = control-lyapunov-function based
# Identification = Joint PD for Identification

sim_time = 20
dt = 5e-3
target_realtime_rate = 1.0

show_diagram = False
make_plots = True

#####################################################

# Drake only loads things relative to the drake path, so we have to do some hacking
# to load an arbitrary file
robot_description_path = "./models/mini_cheetah/mini_cheetah_mesh.urdf"
drake_path = getDrakePath()
robot_description_file = "drake/" + os.path.relpath(robot_description_path, start=drake_path)

robot_urdf = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())

plant = builder.AddSystem(MultibodyPlant(time_step=dt))
assert isinstance(plant, MultibodyPlant)
plant.RegisterAsSourceForSceneGraph(scene_graph)
quad = Parser(plant=plant).AddModelFromFile(robot_urdf, "quad")
quad_frame = plant.GetFrameByName("body")
X_dog = RigidTransform(
    RollPitchYaw(0, 0, 0).ToRotationMatrix(),
    [0, 0, 1.5]
)
plant.WeldFrames(plant.world_frame(), quad_frame, X_dog)

# Add a flat ground with friction
X_BG = RigidTransform()
surface_friction = CoulombFriction(
    static_friction=1.0,
    dynamic_friction=1.0)
plant.RegisterCollisionGeometry(
    plant.world_body(),  # the body for which this object is registered
    X_BG,  # The fixed pose of the geometry frame G in the body frame B
    HalfSpace(),  # Defines the geometry of the object
    "ground_collision",  # A name
    surface_friction)  # Coulomb friction coefficients
plant.RegisterVisualGeometry(
    plant.world_body(),
    X_BG,
    HalfSpace(),
    "ground_visual",
    np.array([0.5, 0.5, 0.5, 0.0]))  # Color set to be completely transparent

# Turn off gravity
# g = plant.mutable_gravity_field()
# g.set_gravity_vector([0,0,0])

plant.Finalize()
assert plant.geometry_source_is_registered()

# Add custom visualizations for the trunk model
trunk_source = scene_graph.RegisterSource("trunk")
trunk_frame = GeometryFrame("trunk")
scene_graph.RegisterFrame(trunk_source, trunk_frame)

trunk_shape = Box(0.4, 0.2, 0.1)
trunk_color = np.array([0.1, 0.1, 0.1, 0.4])
X_trunk = RigidTransform()
X_trunk.set_translation(np.array([0.0, 0.0, 0.0]))

trunk_geometry = GeometryInstance(X_trunk, trunk_shape, "trunk")
if show_trunk_model:
    trunk_geometry.set_illustration_properties(MakePhongIllustrationProperties(trunk_color))
scene_graph.RegisterGeometry(trunk_source, trunk_frame.id(), trunk_geometry)

trunk_frame_ids = {"trunk": trunk_frame.id()}

for foot in ["lf", "rf", "lh", "rh"]:
    foot_frame = GeometryFrame(foot)
    scene_graph.RegisterFrame(trunk_source, foot_frame)

    foot_shape = Sphere(0.02)
    X_foot = RigidTransform()
    foot_geometry = GeometryInstance(X_foot, foot_shape, foot)
    if show_trunk_model:
        foot_geometry.set_illustration_properties(MakePhongIllustrationProperties(trunk_color))

    scene_graph.RegisterGeometry(trunk_source, foot_frame.id(), foot_geometry)
    trunk_frame_ids[foot] = foot_frame.id()

for body_name in ["body", "abduct_fl", "thigh_fl", "shank_fl"]:
    AddMultibodyTriad(plant.GetFrameByName(body_name), scene_graph)

# Create high-level trunk-model planner and low-level whole-body controller
if planning_method == "basic":
    planner = builder.AddSystem(BasicTrunkPlanner(trunk_frame_ids))
elif planning_method == "towr":
    planner = builder.AddSystem(TowrTrunkPlanner(trunk_frame_ids))
elif planning_method == "leg_sine":
    planner = builder.AddSystem(LegSineTrunkPlanner(trunk_frame_ids))
else:
    print("Invalid planning method %s" % planning_method)
    sys.exit(1)

if control_method == "B":
    controller = builder.AddSystem(BasicController(plant, dt, use_lcm=use_lcm))
elif control_method == "ID":
    controller = builder.AddSystem(IDController(plant, dt, use_lcm=use_lcm))
elif control_method == "MPTC":
    controller = builder.AddSystem(MPTCController(plant, dt, use_lcm=use_lcm))
elif control_method == "PC":
    controller = builder.AddSystem(PCController(plant, dt, use_lcm=use_lcm))
elif control_method == "CLF":
    controller = builder.AddSystem(CLFController(plant, dt, use_lcm=use_lcm))
elif control_method == "Identification":
    controller = builder.AddSystem(IdentificationController(plant, dt, use_lcm=use_lcm))
else:
    print("Invalid control method %s" % control_method)
    sys.exit(1)

# Set up the Scene Graph
builder.Connect(
    scene_graph.get_query_output_port(),
    plant.get_geometry_query_input_port())
builder.Connect(
    plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(plant.get_source_id()))
builder.Connect(
    planner.GetOutputPort("trunk_geometry"),
    scene_graph.get_source_pose_port(trunk_source))  # 发送planner可视化几何的位置和姿态

# Connect the trunk-model planner to the controller
if not control_method == "B":
    builder.Connect(planner.GetOutputPort("trunk_trajectory"), controller.get_input_port(1))

# Connect the controller to the simulated plant
builder.Connect(controller.GetOutputPort("quad_torques"),
                plant.get_actuation_input_port(quad))
builder.Connect(plant.get_state_output_port(),
                controller.GetInputPort("quad_state"))

# Add loggers
logger = LogVectorOutput(controller.GetOutputPort("output_metrics"), builder)
logger_accel = LogVectorOutput(plant.get_generalized_acceleration_output_port(), builder)

# Set up the Visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Compile the diagram: no adding control blocks from here on out
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# Visualize the diagram
if show_diagram:
    plt.figure(figsize=(120, 80))
    plot_system_graphviz(diagram, max_depth=2)
    plt.show()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
if use_lcm:
    # If we're using LCM to send messages to another simulator or a real
    # robot, we don't want Drake to slow things down, so we'll publish
    # new messages as fast as possible
    simulator.set_target_realtime_rate(0.0)
else:
    simulator.set_target_realtime_rate(target_realtime_rate)

# Set initial states
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
# q0 = np.asarray([1.0, 0.0, 0.0, 0.0,  # base orientation
#                  0.0, 0.0, 0.3,  # base position
#                  0.0, -0.8, 1.6,
#                  0.0, -0.8, 1.6,
#                  0.0, -0.8, 1.6,
#                  0.0, -0.8, 1.6])
q0 = np.asarray([0.0, -0.8, 1.6,
                 0.0, -0.8, 1.6,
                 0.0, -0.8, 1.6,
                 0.0, -0.8, 1.6])
qd0 = np.zeros(plant.num_velocities())
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, qd0)

# Run the simulation!
simulator.AdvanceTo(sim_time)

if make_plots:
    log = logger.FindLog(diagram_context)
    log_accel = logger_accel.FindLog(diagram_context)
    # Plot stuff
    t = log.sample_times()
    io.savemat("./Problem1/time.mat", {"t": t})
    q_qd_torque = log.data()
    io.savemat("./Problem1/q_qd_torque.mat", {"q_qd_torque": q_qd_torque})
    qdd = log_accel.data()
    io.savemat("./Problem1/qdd.mat", {"qdd": qdd})

    t = log.sample_times()[1:]
    p_feet_desired = log.data()[0, 1:]
    pd_feet_desired = log.data()[12, 1:]
    plt.figure()
    # plt.subplot(4,1,1)
    # plt.plot(t, res, linewidth='2')
    # plt.ylabel("Residual")

    plt.subplot(3, 1, 1)
    plt.plot(t, p_feet_desired, linewidth='2')
    plt.plot(t, pd_feet_desired, linewidth='2')
    plt.axhline(0, linestyle='dashed', color='grey')
    plt.ylabel("$p_desired$")

    # plt.subplot(3, 1, 2)
    # plt.plot(t, pd_feet_desired, linewidth='2')
    # plt.ylabel("$pd_desired$")
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(t, err, linewidth='2')
    # plt.ylabel("$\|y_1-y_2\|^2$")
    # plt.xlabel("time (s)")

    plt.show()
