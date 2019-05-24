import math
from pgmpy.models import BayesianModel
from scipy.stats import uniform
from gigan.core import cascading_resimulation_mh
import numpy as np
import cv2
from gigan.utils import data_load
from gigan.visualization import video
from gigan.extensions.pgmpy import RandomChoice

# ----------------------------------------------------------------------------------------------------------------------
# Program settings: edit only this section
# ----------------------------------------------------------------------------------------------------------------------
goal = (0.5, 0.5)
# map
start = (0.1, 0.1)
iterations = 10000
#sequence_path = "./datasets/ewap_dataset_full/ewap_dataset/seq_eth/"
# video_file = sequence_path + "seq_eth.avi"
#skip_frames = 760
#frame_title = "SEQ_ETH"
sequence_path = "./datasets/ewap_dataset_full/ewap_dataset/seq_hotel/"
video_file = sequence_path + "seq_hotel.avi"
skip_frames = 0
frame_title = "SEQ_HOTEL"
# ----------------------------------------------------------------------------------------------------------------------
Hfile = sequence_path + "H.txt"
obsfile = sequence_path + "obsmat.txt"

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error opening video stream or file: " + video_file)
    exit()
print("Opening video file: " + video_file)
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount/fps
print('FPS = ' + str(fps))
print('Number of frames = ' + str(frameCount))
minutes = int(duration/60)
seconds = int(duration%60)
spf = 1 / fps
mspf = int(spf * 1000)
print('Duration = ' + str(minutes) + ':' + str(seconds) + ' (' + str(duration) + ' seconds)')

if False:
    agentsData, statistics, peds_in_frame = data_load.mil_to_pixels(sequence_path)
    # Parse pedestrian annotations.
    Hfile = sequence_path + "H.txt"
    obsfile = sequence_path + "obsmat.txt"
    # Parse homography matrix.
    H = np.loadtxt(Hfile)
    Hinv = np.linalg.inv(H)
    recorded_frames, peds_in_frame, agents = data_load.parse_annotations(Hinv, obsfile)

    frame_id = 0
    t_curr = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame_id = frame_id + 1
            if frame_id <= skip_frames:
                continue
            # Draw pedestrians
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
                            if truth_frames[p_curr-1] <= frame_id < truth_frames[p_curr]:
                                video.draw_text(frame, (int(prev[1]), int(prev[0])), str(i))
                            cv2.line(frame, p1, p2, color_crossline, 1, cv2.LINE_AA)  # crossline
                            cv2.line(frame, loc1, loc2, color_line, 1, cv2.LINE_AA)
                            prev = curr
                            p_curr = p_curr + 1
                if frame_id == recorded_frames[t_curr + 1]:
                    t_curr = t_curr + 1

            # Draw on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            topLeftCornerOfText = (10, 20)
            fontScale = 0.5
            fontColor = (255, 255, 255)
            lineType = 1

            cv2.putText(frame, 'frame ' + str(frame_id),
                        topLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            # Display the resulting frame
            cv2.imshow(frame_title, frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(mspf) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


data = [] # load from dataset


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

# HOW DO THEY COMPUTE LIKELIHOOD (TRACKTABLE)
start_node.likelihood = None
goal_node.likelihood = None
path_node.likelihood = None
noisy_path_node.likelihood = noisy_path_likelihood

# LIKELIHOOD-FREE (NOT TRACKTABLE)
# TODO: callable to GAN sampler.
#  Start, World and Goal affect path non-explicitly, as they are passed as arguments to a path generator.
#  In this case RRT was used by Cusumano https://arxiv.org/abs/1704.04977
path_node.sample_pdf = None

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

# Now defining the parameters.
print(cusumano_model.get_roots())
print(cusumano_model.get_children(start_node))
print(cusumano_model.get_leaves())
print(cusumano_model.get_parents(noisy_path_node))
# FIXME: not working
# print(cusumano_model.get_independencies())

# Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return accept < (np.exp(x_new - x))


cascading_resimulation_mh(cusumano_model, iterations, data, acceptance)
