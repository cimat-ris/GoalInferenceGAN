import cv2
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from scipy.stats import uniform
from gigan.utils import data_load
from gigan.visualization import video
from gigan.extensions.pgmpy import RandomChoice
from gigan.core import cascading_resimulation_mh


# ----------------------------------------------------------------------------------------------------------------------
# Program settings: edit only this section
# ----------------------------------------------------------------------------------------------------------------------
goal = (0.5, 0.5)
# map
start = (0.1, 0.1)
iterations = 10000
visualization = False
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
def uniform_pdf(u, v):
    return uniform.pdf(x=[u, v], loc=(0,0), scale=(width, height))


def noisy_path_likelihood():
    val = 0
    likelihood = math.exp(-0.5)
    return likelihood


# NODE DEFINITIONS
# Nodes can be any hashable python object. Using GIGAN RandomChoice object
start_node = RandomChoice(name="start")
goal_node = RandomChoice(name="goal")
path_node = RandomChoice(name="path")
noisy_path_node = RandomChoice(name="noisy_path")

# PROPOSAL DISTRIBUTIONS
start_node.proposed_pdf = uniform_pdf
goal_node.proposed_pdf = uniform_pdf
path_node.proposed_pdf = None
noisy_path_node.proposed_pdf = None

# HOW DO THEY COMPUTE LIKELIHOOD (TRACTABLE)
start_node.likelihood = None
goal_node.likelihood = None
path_node.likelihood = None
noisy_path_node.likelihood = noisy_path_likelihood

# LIKELIHOOD-FREE (NOT TRACTABLE)
# TODO: callable to GAN sampler.
#  Start, World and Goal affect path non-explicitly, as they are passed as arguments to a path generator.
#  In this case RRT was used by Cusumano https://arxiv.org/abs/1704.04977
path_node.transition_model = None

# INITIAL OBSERVATIONS
start_node.samples = []
goal_node.samples = []
path_node.samples = []
noisy_path_node.samples = []

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
# Solve and prepare/analyze results
# ----------------------------------------------------------------------------------------------------------------------
if False:
    cascading_resimulation_mh(cusumano_model, iterations, data, acceptance)


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
                    prev = curr
                    p_curr = p_curr + 1
        if frame_id == recorded_frames[t_curr + 1]:
            t_curr = t_curr + 1


if visualization:
    video.video_player(vc, visualization, frame_title, skip_frames, draw_etz)
video.video_close(vc)