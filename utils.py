import pandas as pd
import numpy as np
import time
from itertools import product

def get_file_name(v, h, t_id, s_id):
    vertical = ['Top', 'Middle','Bottom']
    horizontal = ['Left', 'Center', 'Right']

    file_name = 'Position'
    file_name += vertical[v] + horizontal[h] 
    file_name += 'Trajectory' + str(t_id) 
    file_name += 'Subject' + str(s_id) + '.csv'
    return file_name

def load_trajs(file_names):
    trajs = []
    for f in file_names:
        try:
            trajs.append(pd.read_csv('./RightWristData/' + f).dropna().to_numpy())
        except FileNotFoundError:
            pass
    return trajs

def find_max_duration(trajs):
    all_length = []
    for traj in trajs:
        length = len(traj)
        all_length.append(length)
    max_length = np.max(all_length)
    return max_length

def extend_traj(trajs):
    max_length = find_max_duration(trajs)
    extended_trajs = []

    for traj in trajs:
        new = traj.copy()
        extended_trajs.append(new)
    
    for i, traj in enumerate(extended_trajs):              
        last_row = traj[-2:-1, :] 
        while(len(traj) != max_length):
            traj = np.vstack((traj, last_row))
        extended_trajs[i] = traj
    
    return extended_trajs

def visualizeJointTraj(traj, viz, sleep_time=0.00125):
    for n in range(len(traj)):
        viz.display(traj[n])
        time.sleep(sleep_time)

def getExtendedTraj(position_verticle, position_horizontal, selected_file_name, config):
    vertical = config['vertical']
    horizontal = config['horizontal']
    trajectory_ids = config['traj_ids']
    subject_ids = config['subject_ids']
                         
    file_names = [get_file_name(position_verticle, position_horizontal, t, s) for t, s in product(trajectory_ids, subject_ids)]
    trajs = load_trajs(file_names)
    extended = extend_traj(trajs)
    fileNames = {}
    i = 0
    
    for traj_id in range(1, 3):
        for subject_id in range(2, 5):
            file_name = 'Position'
            file_name += vertical[position_verticle] + horizontal[position_horizontal] 
            file_name += 'Trajectory' + str(traj_id) 
            file_name += 'Subject' + str(subject_id)
            fileNames[file_name] = extended[i]
            i += 1
            
    traj = fileNames[selected_file_name]
    
    return traj