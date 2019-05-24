import os
import random
import numpy as np


def mil_to_pixels(directory=["./data/ewap_dataset/seq_hotel"]):
    '''
    Preprocess the frames from the datasets.
    Convert values to pixel locations from millimeters
    obtain and store all frames data the actually used frames (as some are skipped),
    the ids of all pedestrians that are present at each of those frames and the sufficient statistics.
    '''
    def collect_stats(agents):
        x_pos = []
        y_pos = []
        for agent_id in range(1, len(agents)):
            trajectory = [[] for _ in range(3)]
            traj = agents[agent_id]
            for step in traj:
                x_pos.append(step[1])
                y_pos.append(step[2])
        x_pos = np.asarray(x_pos)
        y_pos = np.asarray(y_pos)
        # takes the average over all points through all agents
        return [[np.min(x_pos), np.max(x_pos)], [np.min(y_pos), np.max(y_pos)]]

    Hfile = os.path.join(directory, "H.txt")
    obsfile = os.path.join(directory, "obsmat.txt")
    # Parse homography matrix.
    H = np.loadtxt(Hfile)
    Hinv = np.linalg.inv(H)
    # Parse pedestrian annotations.
    frames, pedsInFrame, agents = parse_annotations(Hinv, obsfile)
    # collect mean and std
    statistics = collect_stats(agents)
    norm_agents = []
    # collect the id, normalised x and normalised y of each agent's position
    pedsWithPos = []
    for agent in agents:
        norm_traj = []
        for step in agent:
            _x = (step[1] - statistics[0][0]) / (statistics[0][1] - statistics[0][0])
            _y = (step[2] - statistics[1][0]) / (statistics[1][1] - statistics[1][0])
            norm_traj.append([int(frames[int(step[0])]), _x, _y])

        norm_agents.append(np.array(norm_traj))

    return np.array(norm_agents), statistics, pedsInFrame


def parse_annotations(Hinv, obsmat_txt):
    """
    Parse the dataset and convert to image frames data.
    :param Hinv:
    :param obsmat_txt:
    :return: recorded_frames (maps timestep -> (first) frame),
             peds_in_frame (maps timestep -> ped IDs),
             peds (maps ped ID -> (t,x,y,z) path)
    """
    def to_image_frame(loc):
        """
        Given H^-1 and (x, y, z) in world coordinates,
        returns (u, v, 1) in image frame coordinates.
        """
        loc = np.dot(Hinv, loc)  # to camera frame
        return loc / loc[2]  # to pixels (from millimeters)

    mat = np.loadtxt(obsmat_txt)
    num_peds = int(np.max(mat[:, 1])) + 1
    peds = [np.array([]).reshape(0, 4) for _ in range(num_peds)]  # maps ped ID -> (t,x,y,z) path

    num_frames = (mat[-1, 0] + 1).astype("int")
    num_unique_frames = np.unique(mat[:, 0]).size
    recorded_frames = [-1] * num_unique_frames  # maps timestep -> (first) frame
    peds_in_frame = [[] for _ in range(num_unique_frames)]  # maps timestep -> ped IDs

    frame = 0
    time = -1
    blqk = False
    for row in mat:
        if row[0] != frame:
            frame = int(row[0])
            time += 1
            recorded_frames[time] = frame

        ped = int(row[1])

        peds_in_frame[time].append(ped)
        loc = np.array([row[2], row[4], 1])
        loc = to_image_frame(loc)
        loc = [time, loc[0], loc[1], loc[2]]
        peds[ped] = np.vstack((peds[ped], loc))

    return recorded_frames, peds_in_frame, peds