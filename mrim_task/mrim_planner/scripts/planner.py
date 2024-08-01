#!/usr/bin/env python3
"""
Custom planner for multi-robot inspection
@author: P. Petracek, V. Kratky, R. Penicka, T. Baca
"""

import rospy, rospkg
import numpy as np

from utils import *

from solvers.tsp_solvers import *
from trajectory import Trajectory, TrajectoryUtils


def to_homogeneous(points):
    """ Convert Cartesian coordinates to homogeneous coordinates """
    points_homogeneous = [np.append(point, 1) for point in points]
    return points_homogeneous

def compute_plucker_coordinates(p1, p2):
    """ Compute Plücker coordinates for the line through points p1 and p2 """
    L01 = p1[0] * p2[1] - p1[1] * p2[0]
    L02 = p1[0] * p2[2] - p1[2] * p2[0]
    L03 = p1[0] * p2[3] - p1[3] * p2[0]
    L23 = p1[2] * p2[3] - p1[3] * p2[2]
    L31 = p1[3] * p2[1] - p1[1] * p2[3]
    L12 = p1[1] * p2[2] - p1[2] * p2[1]
    
    return np.array([L01, L02, L03, L23, L31, L12])

def compute_lines(points):
    """ Compute lines between each pair of points in homogeneous coordinates (3D) """
    points = [p.point.asArray() for p in points]
    points_homogeneous = to_homogeneous(points)
    lines = []

    for i in range(len(points) - 1):
        p1 = points_homogeneous[i]
        p2 = points_homogeneous[i + 1]
        line = compute_plucker_coordinates(p1, p2)
        lines.append(line)
    
    return lines


def compute_distance(point, line):
    """ Compute the distance from a point to a line in 3D using Plücker coordinates """
    x, y, z, w = to_homogeneous(p.point.asArray())
    L01, L02, L03, L23, L31, L12 = line
    
    # Direction vector of the line
    d = np.array([L23, L31, L12])
    # Moment vector of the line
    m = np.array([L01, L02, L03])
    
    # Point as vector
    p = np.array([x, y, z])
    
    # Distance calculation
    numerator = np.linalg.norm(np.cross(d, p) + m)
    denominator = np.linalg.norm(d)
    
    distance = numerator / denominator
    return distance

def closest_line(point, lines):
    """ Find the closest line to a given point in homogeneous coordinates """
    point = point.point.asArray() 
    point_homogeneous = to_homogeneous(point)
    distances = [compute_distance(point_homogeneous, line) for line in lines]
    min_distance_index = np.argmin(distances)
    return min_distance_index, distances[min_distance_index]


class MrimPlanner:

    ALLOWED_COLLISION_AVOIDANCE_METHODS = ['none', 'delay_2nd_till_1st_UAV_finishes', 'delay_till_no_collisions_occur']

    # # #{ __init__()
    def __init__(self):

        rospy.init_node('mrim_planner', anonymous=True)

        ## | ---------------------- load problem ---------------------- |
        problem_filename = rospy.get_param('~problem/name')
        session_problem = rospy.get_param('~session_problem')

        problem_filepath = rospkg.RosPack().get_path('mrim_resources') + "/problems/" + problem_filename

        problem, log_msg = ProblemLoader().loadProblem(problem_filepath)

        if problem is None:
            rospy.logerr(log_msg)
            rospy.signal_shutdown(log_msg)
            exit(-1)

        ## |  load parameters from ROS custom config (mrim_task/mrim_planner/config/custom_config.yaml)  |
        self._viewpoints_distance    = rospy.get_param('~viewpoints/distance', 3.0)
        self._plot                     = rospy.get_param('~problem/plot', False)
        self._trajectory_dt            = rospy.get_param('~trajectories/dt', 0.2)
        self._smoothing_sampling_step  = rospy.get_param('~path_smoothing/sampling_step', 0.1)
        self._smoothing_distance       = rospy.get_param('~path_smoothing/lookahead_dist', 0.3)
        self._sample_with_stops        = rospy.get_param('~trajectory_sampling/with_stops', True)
        self._global_frame             = rospy.get_param('~global_frame', "gps_origin")
        self._tsp_clustering_method    = rospy.get_param('~tsp/clustering', 'random')
        self._custom_cluster_split     = rospy.get_param('~tsp/custom_cluster_split', 'even')

        max_vel_x                      = rospy.get_param('~dynamic_constraints/max_velocity/x', 1.0)
        max_vel_y                      = rospy.get_param('~dynamic_constraints/max_velocity/y', 1.0)
        max_vel_z                      = rospy.get_param('~dynamic_constraints/max_velocity/z', 1.0)
        max_acc_x                      = rospy.get_param('~dynamic_constraints/max_acceleration/x', 1.0)
        max_acc_y                      = rospy.get_param('~dynamic_constraints/max_acceleration/y', 1.0)
        max_acc_z                      = rospy.get_param('~dynamic_constraints/max_acceleration/z', 1.0)
        self._max_velocity             = (max_vel_x, max_vel_y, max_vel_z)
        self._max_acceleration         = (max_acc_x, max_acc_y, max_acc_z)
        self._max_heading_rate         = rospy.get_param('~dynamic_constraints/max_heading_rate', 1.0)
        self._max_heading_acceleration = rospy.get_param('~dynamic_constraints/max_heading_rate_acceleration', 1.0)

        ## | ---------------- setup collision avoidance --------------- |
        self._safety_distance_mutual = rospy.get_param('~trajectories/min_distance/mutual')
        self._collision_avoidance    = rospy.get_param('~collision_avoidance/method', 'none')

        ## | ------------------- setup path planner ------------------- |
        self._path_planner = {}
        self._path_planner['timeout']                       = rospy.get_param('~path_planner/timeout', 1.0)
        self._path_planner['path_planning_method']          = rospy.get_param('~path_planner/method', 'rrt')
        self._path_planner['safety_distance']               = rospy.get_param('~trajectories/min_distance/obstacles')
        self._path_planner['distance_estimation_method']    = rospy.get_param('~tsp/distance_estimates', 'euclidean')
        self._path_planner['straighten']                    = rospy.get_param('~path_planner/straighten_paths')

        pp_method = self._path_planner['path_planning_method']
        de_method = self._path_planner['distance_estimation_method']
        if pp_method == 'astar' or de_method == 'astar':
            self._path_planner['astar/grid_resolution'] = rospy.get_param('~path_planner/astar/grid_resolution')
        if pp_method.startswith('rrt') or de_method.startswith('rrt'):
            self._path_planner['rrt/branch_size']      = rospy.get_param('~path_planner/rrt/branch_size')
            self._path_planner['rrt/sampling/method']  = rospy.get_param('~path_planner/rrt/sampling/method')
            self._path_planner['rrtstar/neighborhood'] = None

            if pp_method == 'rrtstar' or de_method == 'rrtstar':
                self._path_planner['rrtstar/neighborhood'] = rospy.get_param('~path_planner/rrt/star/neighborhood', None)

            if self._path_planner['rrt/sampling/method'] == 'gaussian':
                self._path_planner['rrt/sampling/gaussian/stddev_inflation']  = rospy.get_param('~path_planner/rrt/sampling/gaussian/stddev_inflation')

        ## | -------------- print out general parameters -------------- |
        # print('using parameters:')
        # print(' viewpoints distance:', self._viewpoints_distance)
        # print(' max velocity:', self._max_velocity)
        # print(' max acceleration:', self._max_acceleration)
        # print(' max heading rate:', self._max_heading_rate)
        # print(' max heading acceleration:', self._max_heading_acceleration)
        # print(' smoothing lookahead distance:', self._smoothing_distance)
        # print(' smoothing sampling step:', self._smoothing_sampling_step)
        # print(' plot:', self._plot)
        # print(' trajectory dT:', self._trajectory_dt)
        # print(' path planning method:', pp_method)
        # print(' distance estimation method:', de_method)

        ## | ----------------- initiate ROS publishers ---------------- |
        self.publisher_trajectory_1 = rospy.Publisher("~trajectory_1_out", TrajectoryReference, queue_size=1, latch=True)
        self.publisher_trajectory_2 = rospy.Publisher("~trajectory_2_out", TrajectoryReference, queue_size=1, latch=True)
        self.problem_publisher      = rospy.Publisher("~problem_out", InspectionProblem, queue_size=1, latch=True)

        rate = rospy.Rate(1.0)
        rate.sleep()

        ## | --------------------- publish problem -------------------- |
        self.problem_publisher.publish(problem)

        ## | -------------------- plan trajectories ------------------- |
        trajectories, plotter = self.planTrajectories(problem)

        # # | -------------- convert to ROS trajectories -------------- |
        ros_trajectory_1 = trajectoryToRosMsg(trajectories[0].getPoses(), self._global_frame)
        ros_trajectory_2 = trajectoryToRosMsg(trajectories[1].getPoses(), self._global_frame)

        ## | ------------------ publish trajectories ------------------ |
        self.publisher_trajectory_1.publish(ros_trajectory_1)
        self.publisher_trajectory_2.publish(ros_trajectory_2)

        plotter.show(legend=True)

        rospy.loginfo('Trajectories published, staying on, the publishers are latched.')
        rospy.spin()
    # # #}

    # # #{ planTrajectories()
    def planTrajectories(self, problem):

        ## | --------------- create visualization object -------------- |
        plotter = ProblemPlotter(self._plot)
        plotter.addProblem(problem)

        ## | ----- initialize objects for TSP and trajectory utils ---- |
        tsp_solver       = TSPSolver3D()
        trajectory_utils = TrajectoryUtils(self._max_velocity, self._max_acceleration, self._trajectory_dt)

        # # #{ Cluster target locations

        print('[ASSIGNING VIEWPOINTS TO UAVs]')

        viewpoints       = []
        nonclustered_vps = []

        for r in range(problem.number_of_robots):

            # add starting pose of the robot
            start_vp = Viewpoint(0, Pose(problem.start_poses[r].position.x, problem.start_poses[r].position.y, problem.start_poses[r].position.z, problem.start_poses[r].heading))
            viewpoints.append([start_vp])

            # get robot ID
            robot_id = problem.robot_ids[r]
            for ip in problem.inspection_points:

                # convert IP to VP [id x y z heading]
                viewpoint = inspectionPointToViewPoint(ip, self._viewpoints_distance)

                # if inspectability of IP is unique for robot with this ID, add it
                if len(ip.inspectability) == 1 and robot_id in ip.inspectability:
                    viewpoints[r].append(viewpoint)

                # if inspectability of IP is arbitrary, store it for clustering
                elif len(ip.inspectability) != 1 and ip.idx not in [nips.idx for nips in nonclustered_vps]:
                    nonclustered_vps.append(viewpoint)

        # Cluster the rest of the viewpoints into two separate groups
        ## MODIFIED ##
        # 
        mean_position = dict()
        clusters = dict()
        for r in range(problem.number_of_robots):
            clusters[r] = []
            mean_position[r] = np.mean(np.array([vp.pose.point.asList() for vp in viewpoints[r]]), axis=0)

        print(mean_position)

        vps_closest_order = dict()
        vps_distances = dict()
        for r in range(problem.number_of_robots):
            vps_distances[r] = dict()
            for point in nonclustered_vps:
                vps_distances[r][point.idx] = np.linalg.norm( mean_position[r] - point.pose.point.asArray())
            vps_closest_order[r] = nonclustered_vps.copy()
            vps_closest_order[r].sort(key=lambda x: vps_distances[r][x.idx])
        #     print(nonclustered_vps)
        #     print(vps_closest_order[r] )

        
        # for r in range(problem.number_of_robots):
        #     print(len(vps_closest_order[r]))
        if self._tsp_clustering_method == 'close_means':
            if self._custom_cluster_split == 'even':
                sorted_vps = []
                print(len(sorted_vps))
                print(len(nonclustered_vps))
                while len(sorted_vps) < len(nonclustered_vps):
                    for r in range(problem.number_of_robots):
                        if len(sorted_vps) >= len(nonclustered_vps):
                            break
                        for point in vps_closest_order[r]:
                            if point not in sorted_vps:
                                next_closest = point
                                break
                        clusters[r].append(next_closest)
                        sorted_vps.append(next_closest)
                        # print(str(r)+": ")

            elif self._custom_cluster_split == 'closest':
                print(len(viewpoints[1]) > 2 or len(viewpoints[0]) > 2)
                print(viewpoints)
                if len(viewpoints[1]) > 2 or len(viewpoints[0]) > 2:
                    for point in nonclustered_vps:
                        closest_mean = np.inf
                        robot = None
                        for r in range(problem.number_of_robots):
                            if closest_mean > vps_distances[r][point.idx]:
                                closest_mean = closest_mean > vps_distances[r][point.idx]
                                robot = r
                        clusters[robot].append(point)
                else:
                    clusters = tsp_solver.clusterViewpoints(problem, nonclustered_vps, method=self._tsp_clustering_method)

        elif self._tsp_clustering_method == 'kmeans' or self._tsp_clustering_method == 'random':
            clusters = tsp_solver.clusterViewpoints(problem, nonclustered_vps, method=self._tsp_clustering_method)
        elif self._tsp_clustering_method == 'path_clustering':
            while len(nonclustered_vps) > 0:
                segments = dict()
                best_robot = None
                shortest_dist = 9999999999999
                point = None
                for r in range(problem.number_of_robots):
                    tour = tsp_solver.plan_tour(problem, viewpoints[r], self._path_planner)
                    segments[r] = compute_lines(tour)
                    
                for p in nonclustered_vps:
                    for r in range(problem.number_of_robots):
                        _, dist = closest_line(p, segments[r])
                        if dist < shortest_dist:
                            point = p
                            best_robot = r
                            shortest_dist = dist

                viewpoints[best_robot].append(point)
                nonclustered_vps.remove(point)
        


        print(clusters)

        if self._tsp_clustering_method != 'path_clustering':
            for r in range(problem.number_of_robots):
                viewpoints[r].extend(clusters[r])

        ## END MODIFIED ##

        # print out viewpoints
        for i in range(len(viewpoints)):
            print('viewpoints for UAV:', problem.robot_ids[i])
            for vp in viewpoints[i]:
                print('   [{:d}]:'.format(vp.idx), vp.pose)

        # add VPs to offline visualization
        plotter.addViewPoints(viewpoints, self._viewpoints_distance, self._viewpoints_distance)

        # # #}

        # Print out if the viewpoints collide with the environment
        for i in range(len(viewpoints)):
            for vp in viewpoints[i]:
                point = vp.pose.point
                if pointCollidesWithObstacles(point, [Point(o.x, o.y, o.z) for o in problem.obstacle_points], self._path_planner['safety_distance']):
                    rospy.logwarn('VP at %s collides with obstacles.', point)
        # plotter.show(legend=True)

        # # #{ Solve TSP to obtain waypoint path
        print('[PLANNING TSP TOUR]')

        waypoints = []
        for i in range(problem.number_of_robots):

            ## | --------------- Plan tour with a TSP solver -------------- |
            robot_waypoints = tsp_solver.plan_tour(problem, viewpoints[i], self._path_planner) # find decoupled TSP tour over viewpoints
            waypoints.append(robot_waypoints)

            ## | ------------- add waypoints to visualization ------------- |
            plotter.addWaypoints(robot_waypoints, color=COLORS[i], lw=1.2, label='traj (id: ' + str(problem.robot_ids[i]) + ')')
        # # #}

        # # #{ Sample waypoints to trajectories
        trajectories     = []

        # create dynamic constraints
        constraints_velocity     = [self._max_velocity[0], self._max_velocity[1], self._max_velocity[2], self._max_heading_rate] # per axis velocity limits
        constraints_acceleration = [self._max_acceleration[0], self._max_acceleration[1], self._max_acceleration[2], self._max_heading_acceleration] # per axis acceleration limits

        ## | ------------------- Sample waypoints ------------------ |
        print("[PRE COLLISION AVOIDANCE] for robot with ID: {:d}".format(problem.robot_ids[r]))

        # for each robot
        for r in range(problem.number_of_robots):

            # generate trajectory for the robot's VPs
            print('[GENERATING TRAJECTORY]')
            trajectory = Trajectory(self._trajectory_dt, waypoints[r])

            # sample trajectory through its waypoints
            print("[SAMPLING TRAJECTORY]")
            trajectory = trajectory_utils.sampleTrajectoryThroughWaypoints(trajectory, with_stops=self._sample_with_stops,\
                                                                           smooth_path=True, smoothing_la_dist=self._smoothing_distance,\
                                                                           smoothing_sampling_step=self._smoothing_sampling_step,\
                                                                           velocity_limits=constraints_velocity,
                                                                           acceleration_limits=constraints_acceleration)

            if trajectory is None:
                rospy.logerr('Unable to sample trajectory through waypoints. Read the log output to find out why.')
                rospy.signal_shutdown('Unable to sample trajectory through waypoints. Read the log output to find out why.');
                exit(-3)

            trajectories.append(trajectory)
        # # #}
        # print("==========================================segments=========================================")
        segments = trajectory_utils.computeCollisionSegmentsOfTwoTrajectories(trajectories[0], trajectories[1], self._safety_distance_mutual)
        print(segments)
        
        
        ## | ------------------- Resolve collisions ------------------- |
        if self._collision_avoidance in self.ALLOWED_COLLISION_AVOIDANCE_METHODS:
            trajectories, delayed_robots, delays = trajectory_utils.resolveCollisions(self._collision_avoidance, problem, trajectories, self._safety_distance_mutual)
        else:
            print("[COLLISION AVOIDANCE] unknown method: {:s}".format(self._collision_avoidance))

        ## | ------ Add trajectories to the offline visualization ----- |
        for i in range(problem.number_of_robots):
            plotter.addTrajectoryPoses(trajectories[i].getPoses(), color=COLORS[i], label='traj. samples (id: ' + str(problem.robot_ids[i]) + ')')

        # # #{ Print trajectory infos
        print('###############################')
        print('##### Output trajectories #####')
        print('###############################')
        traj_t_max_idx, traj_d_max_idx = np.argmax([t.getTime() for t in trajectories]), np.argmax([t.getLength() for t in trajectories])
        for r in range(problem.number_of_robots):
            print('UAV ID: {:d}'.format(problem.robot_ids[r]))
            print('   Number of VPs:   {:d}'.format(len(viewpoints[r])-1))
            postfix  = ' (max)' if r == traj_t_max_idx else ''
            if r in delayed_robots:
                idx = delayed_robots.index(r)
                postfix += ' (incl. {:.2f} s delay)'.format(delays[idx])
            print('   Trajectory time: {:0.2f} s{:s}'.format(trajectories[r].getTime(), postfix))
            print('   Trajectory len:  {:0.2f} m{:s}'.format(trajectories[r].getLength(), ' (max)' if r == traj_d_max_idx else ''))
        print('###############################')
        # # #}

        # # | --------------- plot velocity profiles --------------- |
        # plotter.plotDynamics(trajectories, self._max_velocity, self._max_acceleration, problem.robot_ids, dt=trajectory.dT)

        return trajectories, plotter
    # # #}

if __name__ == '__main__':
    try:
        mrim_planner = MrimPlanner()
    except rospy.ROSInterruptException:
        pass
