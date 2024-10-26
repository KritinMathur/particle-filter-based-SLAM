import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError
        x_clip, y_clip = np.clip(x,s.xmin,s.xmax),np.clip(y,s.ymin,s.ymax)
        x_index = (np.ceil((x_clip - s.xmin)/s.resolution)).astype(int)
        y_index = (np.ceil((y_clip - s.ymin)/s.resolution)).astype(int)
        return np.vstack((x_index, y_index))

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        n = p.shape[1]
        new_particles = np.zeros_like(p)
        new_weights = np.zeros_like(w)
        r = np.random.uniform(0, 1/n)
        c = w[0]
        i = 0
        for m in range(n):
            u = r + m/n
            while u > c:
                i += 1
                c += w[i]
            new_particles[:, m] = p[:, i]
            new_weights[m] = 1/n
        return new_particles, new_weights

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        d_filter = np.minimum(np.maximum(d, s.lidar_dmin), s.lidar_dmax)
        # 1. from lidar distances to points in the LiDAR frame
        pt_lidar_2D = np.vstack((d_filter*np.cos(angles), d_filter*np.sin(angles)))
        pt_lidar_2D_homo = make_homogeneous_coords_2d(pt_lidar_2D)
        pt_lidar_3D_homo = make_homogeneous_coords_3d(pt_lidar_2D_homo)
        # 2. from LiDAR frame to the body frame
        v_lidar_to_body = np.array([0,0,s.lidar_height])
        T_lidar_to_body = euler_to_se3(0,head_angle,neck_angle,v_lidar_to_body)
        pt_body = T_lidar_to_body @ pt_lidar_3D_homo
        # 3. from body frame to world frame
        v_body_to_world = np.array([p[0],p[1],s.head_height])
        T_body_to_world = euler_to_se3(0,0,p[2],v_body_to_world)
        pt_world_homo = T_body_to_world @ pt_body
        pt_world = pt_world_homo[:2]
        return pt_world

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError

        p2 = s.lidar[t]['xyth']
        p1 = s.lidar[t-1]['xyth']
        return smart_minus_2d(p2,p1)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError

        control = s.get_control(t)
        
        new_particles = np.zeros_like(s.p)
        for i in range(s.n):
            particle = s.p[:,i]
            noisy_control = np.random.multivariate_normal(mean=control, cov=s.Q)
            new_particle = smart_plus_2d(particle,noisy_control)
            new_particles[:,i] = new_particle
        s.p = new_particles.copy()

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError
        new_w_log = np.log(w) + obs_logp
        new_w_normalized =  np.exp(new_w_log - slam_t.log_sum_exp(new_w_log))

        return new_w_normalized

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError

        lidar_timestamp = s.lidar[t]['t']
        joint_timestamp_idx = s.find_joint_t_idx_from_lidar(lidar_timestamp)
        neck_angle,head_angle = s.joint['head_angles'][:,joint_timestamp_idx]
        lidar_distances = s.lidar[t]['scan']
        obs_logp = np.zeros(s.n)
        for i in range(s.n):
            particle = s.p[:,i]
            pt_world = s.rays2world(particle, lidar_distances, head_angle, neck_angle, s.lidar_angles)
            xy_index = s.map.grid_cell_from_xy(pt_world[0], pt_world[1])
            obs_logp[i] = np.sum(s.map.cells[xy_index[0],xy_index[1]])
        new_weights = s.update_weights(s.w,obs_logp)
        s.w = new_weights.copy()
        
        particle_idx_max_w = np.argmax(s.w)
        s.best_particle = s.p[:,particle_idx_max_w]
        occ_pt_world = s.rays2world(s.best_particle, lidar_distances, head_angle, neck_angle, s.lidar_angles)
        occ_xy_index = s.map.grid_cell_from_xy(occ_pt_world[0], occ_pt_world[1])

        # Update for Occ cells
        s.map.log_odds[occ_xy_index[0],occ_xy_index[1]] += s.lidar_log_odds_occ

        # Ray Tracing for Free cells
        all_xy = []

        for i in range(occ_xy_index.shape[1]):
            x1, y1 = s.best_particle[0],s.best_particle[1]
            x2, y2 = pt_world[:,i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            num_intervals = int(distance / s.map.resolution)
            x_values = np.linspace(x1, x2, num_intervals)[1:-1]
            y_values = np.linspace(y1, y2, num_intervals)[1:-1]
            xy_free = s.map.grid_cell_from_xy(x_values,y_values)
            all_xy.append(xy_free)
        
        all_xy_dupes = np.hstack(all_xy)
        unique_xy_tuples = {tuple(column) for column in all_xy_dupes.T}
        unique_xy = np.array(list(unique_xy_tuples)).T
        s.map.log_odds[unique_xy[0],unique_xy[1]] += s.lidar_log_odds_free

        #Map binarization
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
        s.map.cells = (s.map.log_odds > s.map.log_odds_thresh).astype(int).copy()    

        

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
