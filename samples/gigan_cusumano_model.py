import cv2
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gigan.utils.geometry as geom
from pgmpy.models import BayesianModel
from scipy.stats import uniform, randint
from gigan.utils import data_load
from gigan.visualization import video
from gigan.extensions.pgmpy import RandomChoice
from gigan.extensions.sgan import get_eth_gan_generator, sample_generator
from gigan.core import cascading_resimulation_mh


# ----------------------------------------------------------------------------------------------------------------------
# Program settings: edit only this section
# ----------------------------------------------------------------------------------------------------------------------
goal = (0.5, 0.5)
# map
start = (0.1, 0.1)
iterations = 10000
visualization = True
t_curr = 0
#sequence_path = "./datasets/ewap_dataset_full/ewap_dataset/seq_eth/"
# video_file = sequence_path + "seq_eth.avi"
#skip_frames = 760
#frame_title = "SEQ_ETH"
sequence_path = "./datasets/ewap_dataset_full/ewap_dataset/seq_hotel/"
video_file = sequence_path + "seq_hotel.avi"
skip_frames = 0
frame_title = "SEQ_HOTEL"


# ----------------------------------------------------------------------------------------------------------------------
# Dataset loading and parsing
# ----------------------------------------------------------------------------------------------------------------------
agentsData, statistics, peds_in_frame = data_load.mil_to_pixels(sequence_path)
# Parse homography matrix.
H = np.loadtxt(sequence_path + "H.txt")
Hinv = np.linalg.inv(H)
# Parse pedestrian annotations.
recorded_frames, peds_in_frame, agents = data_load.parse_annotations(Hinv, sequence_path + "obsmat.txt")
vc = video.video_handler(video_file)
width, height, frameCount, fps = video.video_properties(vc)
print('FPS = ' + str(fps))
print('Number of frames = ' + str(frameCount))
data = []  # load from dataset


# ----------------------------------------------------------------------------------------------------------------------
# Preparation and Definitions needed for the solution
# ----------------------------------------------------------------------------------------------------------------------
# Likelihood for goal and start positions. This evaluates how likely is
# (u, v) coordinate. Here I model the goal and the start to be a polar
# coordinate. \theta \in (0, 2\pi), discrete space of 360*4.
def uniform_pdf(theta):
    # All angles are equally probable, rendering a likelihood of 0.0006944444444444445
    return uniform.pdf(x=theta, loc=0, scale=1440)


def uniform_sample():
    # An angle represented as four times the 360 degrees. Hence 1440 possible values
    # When using it as radians do [ uniform_sample() * math.pi / 180 * 4 ]
    # If evaluating likelihood use as is.
    return randint.rvs(low=0, high=1439, size=1)


def noisy_path_likelihood(): # noisy_path: Trajectory, path: Trajectory):
    val = 0
    likelihood = math.exp(-0.5)
    return likelihood


# NODE DEFINITIONS
# Nodes can be any hashable python object. Using GIGAN RandomChoice object
start_node = RandomChoice(name="start")
goal_node = RandomChoice(name="goal")
path_node = RandomChoice(name="path")
noisy_path_node = RandomChoice(name="noisy_path")

# EVIDENCES
start_node.observed = True
goal_node.observed = False
path_node.observed = False
noisy_path_node.observed = True

# INITIAL OBSERVATIONS
start_node.samples = []  # To be set when playing video
goal_node.samples = []  # To be set when playing video
path_node.samples = []  # To be set when playing video
noisy_path_node.samples = []  # To be set when playing video

# HOW DO THEY COMPUTE LIKELIHOOD (TRACTABLE)
start_node.likelihood = uniform_pdf
goal_node.likelihood = uniform_pdf
path_node.likelihood = None
noisy_path_node.likelihood = noisy_path_likelihood

# HOW DO THEY RANDOM WALK (PROPOSAL DISTRIBUTIONS)
start_node.transition = uniform_sample
goal_node.transition = uniform_sample

# TESTING FUNCTIONS
theta = uniform_sample()  # sample a goal or a start
print(theta)
print(uniform_pdf(theta)) # likelihood of sampled goal or start

# TODO: callable to GAN sampler.
#  Start, World and Goal affect path non-explicitly, as they are passed as arguments to a path generator.
#  In this case RRT was used by Cusumano https://arxiv.org/abs/1704.04977
generator = get_eth_gan_generator(load_mode="CPU")
obs_traj = None
# sample_generator(generator, obs_traj)
path_node.transition = None
noisy_path_node.transition = None

# CREATING BAYESIAN MODEL (GRAPH)
cusumano_model = BayesianModel([
    (start_node, path_node),
    (goal_node, path_node),
    (path_node, noisy_path_node)
])

# ATTENTION!!! =)
# Test functions. If these do not work you probably need to check python requirements.txt:
# this was tested using networkx 2.3 and pgmpy 0.1.7 from dev branch in github.
print("Root nodes: %s" % cusumano_model.get_roots())
print("Children of 'start': %s" % cusumano_model.get_children(start_node))
print("Leaf nodes: %s" % cusumano_model.get_leaves())
print("Parents of 'noisy_path': %s" % cusumano_model.get_parents(noisy_path_node))
# FIXME: not working: we get error 'RandomChoice' object is not iterable'.
# print(cusumano_model.get_independencies())

nx.draw_networkx(cusumano_model, arrowsize=15, node_size=800, node_color='#90b9f9')
plt.show()

# Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return accept < (np.exp(x_new - x))


# ----------------------------------------------------------------------------------------------------------------------
# Video Visualization and results drawing
# ----------------------------------------------------------------------------------------------------------------------
def draw_etz(frame_id: int, frame: object):
    # Draw pedestrians
    global t_curr
    if recorded_frames[t_curr] <= frame_id <= recorded_frames[t_curr + 1]:
        for i in peds_in_frame[t_curr]:
            truth = []
            truth_frames = []
            for row in agentsData[i]:
                truth_frames.append(row[0])
                _x = int(row[1] * (statistics[0][1] - statistics[0][0]) + statistics[0][0])
                _y = int(row[2] * (statistics[1][1] - statistics[1][0]) + statistics[1][0])
                truth.append((_x, _y))
            truth = np.array(truth)
            truth_frames = np.array(truth_frames)
            # Draw only if min frame and max frame contain frame_id
            if truth_frames.min() <= frame_id <= truth_frames.max():
                # Plot the real trajectory
                color_line = (10, 120, 0)
                color_crossline = (0, 255, 0)
                prev = truth[0]
                p_curr = 0
                for curr in truth[1:]:
                    # --------------------------------------------------------------------------------------------------
                    # INFERENCE
                    # --------------------------------------------------------------------------------------------------
                    # Reset samples
                    global cusumano_model, start_node, goal_node, path_node, noisy_path_node
                    start_node.samples.clear()
                    start_node.samples.append(uniform_sample() * math.pi / (180 * 4))
                    goal_node.samples.clear()
                    goal_node.samples.append(uniform_sample() * math.pi / (180 * 4))
                    path_node.samples.clear()
                    noisy_path_node.samples.clear()
                    noisy_path_node.samples.append(truth)
                    #if len(truth) > 3:
                    #    cascading_resimulation_mh(cusumano_model, iterations, data, acceptance)
                    # --------------------------------------------------------------------------------------------------
                    # DRAWINGS
                    # --------------------------------------------------------------------------------------------------
                    # Draw person start
                    pixel = geom.angle_to_pixel(start_node.samples[0], width, height)
                    cv2.circle(frame, (int(pixel[0]), int(pixel[1])), 5, (0, 0, 255), -1)
                    # Draw person inferred goals
                    # TODO: loop on all goals in the samples vector
                    pixel = geom.angle_to_pixel(goal_node.samples[0], width, height)
                    cv2.circle(frame, (int(pixel[0]), int(pixel[1])), 5, (0, 0, 255), -1)
                    # Draw person generated paths (from SGAN)

                    # Draw person noisy path
                    loc1 = (int(prev[1]), int(prev[0]))  # (y, x)
                    loc2 = (int(curr[1]), int(curr[0]))  # (y, x)
                    p1, p2 = video.crossline(curr, prev, 3)
                    if frame_id < truth_frames[p_curr]:
                        color_line = (0, 100, 255)
                        color_crossline = (0, 230, 255)
                    if truth_frames[p_curr - 1] <= frame_id < truth_frames[p_curr]:
                        video.draw_text(frame, (int(prev[1]), int(prev[0])), str(i))
                    cv2.line(frame, p1, p2, color_crossline, 1, cv2.LINE_AA)  # crossline
                    cv2.line(frame, loc1, loc2, color_line, 1, cv2.LINE_AA)
                    # --------------------------------------------------------------------------------------------------
                    # Loop control
                    prev = curr
                    p_curr = p_curr + 1

        if frame_id == recorded_frames[t_curr + 1]:
            t_curr = t_curr + 1


if visualization:
    video.video_player(vc, frame_title, skip_frames, draw_etz)
video.video_close(vc)