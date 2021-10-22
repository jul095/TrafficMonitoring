#  ****************************************************************************
#  @evaluation_measurement.py
#
#  Evaluate and compare the hole method with ground truth dGPS data
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import argparse
import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from pyproj import Transformer

from run_on_video import process_video

# Constants to convert GPS week and seconds into a utc timestamp
LEAP_SECONDS = 18
FPS_VIDEO = 30


def format_time(timestamp):
    s = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    head = s[:-7]  # everything up to the '.'
    tail = s[-7:]  # the '.' and the 6 digits after it
    f = float(tail)
    temp = "{:.02f}".format(f)  # for Python 2.x: temp = "%.3f" % f
    new_tail = temp[1:]  # temp[0] is always '0'; get rid of it
    return head + new_tail


def gps_week_seconds_to_utc(gpsweek, gpsseconds):
    """
    The gps time starts at 06.01.1980. With the leap seconds a time stamp will be returned
    """
    datetimeformat = "%Y-%m-%d %H:%M:%S.%f"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00.00", datetimeformat)
    elapsed = datetime.timedelta(days=(gpsweek * 7), hours=1, seconds=(gpsseconds - LEAP_SECONDS))
    new_timestamp = epoch + elapsed
    utc_timetamp = datetime.datetime.strftime(new_timestamp, datetimeformat)
    return format_time(new_timestamp)


def get_timestamp_per_frame(frame_id, start_timestamp):
    result_time = start_timestamp + datetime.timedelta(seconds=frame_id / FPS_VIDEO)
    return format_time(result_time)


def prepare_measurement_data(df_measurement):
    # Convert from Rad to Deg
    df_measurement['pos_x'] = df_measurement['pos_x'].apply(lambda x: np.rad2deg(x))
    df_measurement['pos_y'] = df_measurement['pos_y'].apply(lambda x: np.rad2deg(x))

    # Define geographic coordinate transformer
    # xx long, yy Lat
    transformer = Transformer.from_crs("epsg:7931", "epsg:32632", always_xy=True)

    df_measurement[['pos_x', 'pos_y']] = df_measurement[['pos_x', 'pos_y']].apply(
        lambda x: pd.Series(list(transformer.transform(x['pos_x'], x['pos_y']))), axis=1)

    df_measurement['gt_velocity'] = df_measurement[['vel_x', 'vel_y']].apply(
        lambda x: np.sqrt(np.power(x['vel_x'], 2) + np.power(x['vel_y'], 2)) * 3.6, axis=1)

    df_measurement['timestamp'] = df_measurement[['week', 'gps_tow']].apply(
        lambda x: gps_week_seconds_to_utc(x['week'], x['gps_tow']), axis=1)
    res = df_measurement[['pos_x', 'pos_y', 'timestamp']]
    return df_measurement


def get_pixel_coordinate(geo_coordinate, affine, translation):
    """
    get pixel coordinate by translation and affine transformation
    """
    pixel_point = (geo_coordinate - translation) * affine
    pixel_point_x = int(pixel_point[0][0])
    pixel_point_y = int(pixel_point[1][1])
    return pixel_point_x, pixel_point_y


def show_measurement_in_map(df_joined_df, scenario_name):
    """
    Method use a georeferenced map from qgis and a conversion matrix to plot the trajectories into a map
    """
    reference_map = cv2.imread(os.path.join('config', 'crossing_map_referenced_with_camera_image.png'))
    world_file = np.loadtxt(os.path.join('config', 'crossing_map_referenced_with_camera_image.pgw')).reshape((3, 2))
    affine = np.linalg.inv(world_file[0:2])
    translation = world_file[2]
    window_name = f'evaluation trajectories: {scenario_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    pixel_point_measurements, pixel_point_estimations, pixel_point_detection_estimations = [], [], []

    for index, trajectory in df_joined_df.iterrows():
        geo_coordinate_measurement = np.array([trajectory['gt_x'], trajectory['gt_y']])
        pixel_point_measurement = get_pixel_coordinate(geo_coordinate_measurement, affine, translation)

        # cv2.circle(reference_map, pixel_point_measurement, 3, (26, 153, 47),3)

        pixel_point_measurements.append(pixel_point_measurement)

    for index, trajectory in df_joined_df.iterrows():
        geo_coordinate_estimation = np.array([trajectory['estimation_x_opt'], trajectory['estimation_y_opt']])
        detection_coordinate_estimation = np.array([trajectory['estimation_x'], trajectory['estimation_y']])
        pixel_point_estimation = get_pixel_coordinate(geo_coordinate_estimation, affine, translation)
        pixel_point_detection_estimation = get_pixel_coordinate(detection_coordinate_estimation, affine, translation)
        # B G R
        #  overlay = cv2.circle(overlay, pixel_point_estimation, 3, , 3)
        pixel_point_estimations.append(pixel_point_estimation)
        pixel_point_detection_estimations.append(pixel_point_detection_estimation)

    #  for pixel_point_measurement, pixel_point_estimation in zip(pixel_point_measurements, pixel_point_estimations):
    #    cv2.line(overlay,pixel_point_measurement, pixel_point_estimation,(0,0,0),2,cv2.LINE_AA)
    cv2.polylines(reference_map, [np.asarray(pixel_point_measurements, dtype=np.int32)], False, (26, 153, 47), 7)
    overlay = reference_map.copy()
    cv2.polylines(overlay, [np.asarray(pixel_point_estimations, dtype=np.int32)], False, (28, 28, 255), 7)
    cv2.polylines(overlay, [np.asarray(pixel_point_detection_estimations, dtype=np.int32)], False, (230, 28, 28), 7)
    alpha = 0.8
    reference_map = cv2.addWeighted(overlay, alpha, reference_map, 1 - alpha, 0, reference_map)
    cv2.imshow(window_name, reference_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kalman_filter(df_trajectories):
    """
    The Kalman filter offers smoothing of the final trajectories
    But it's not applied in the final evaluation
    """
    measurement = df_trajectories[['coordinate_world_x_opt', 'coordinate_world_y_opt']]

    initial_state_mean = [measurement['coordinate_world_x_opt'].iloc[0],
                          0,
                          measurement['coordinate_world_y_opt'].iloc[1],
                          0]
    measurement = np.asarray(measurement)
    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    observation_covariance = [[10, 0], [0, 10]]
    kf1 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean,
                       observation_covariance=observation_covariance,
                       em_vars=['transition_covariance', 'initial_state_covariance'])

    kf = kf1.em(measurement, n_iter=10)
    # smoothed, _ = kf.smooth(measurement)

    # kf2 = KalmanFilter(transition_matrices=transition_matrix,
    #                   observation_matrices=observation_matrix,
    #                   initial_state_mean=initial_state_mean,
    #                   observation_covariance=100000 * kf1.observation_covariance,
    #                   em_vars=['transition_covariance', 'initial_state_covariance'])

    # kf2 = kf2.em(measurement, n_iter=10)
    (smoothed_measurement, smoothed_state_covariances) = kf.smooth(measurement)

    df_trajectories['coordinate_world_x_opt_kf'] = smoothed_measurement[:, 0]
    df_trajectories['coordinate_world_y_opt_kf'] = smoothed_measurement[:, 2]
    return df_trajectories


def evaluate_one_scenario(scenario_name, is_plot_vis, folder_measurement, folder_matching, folder_trajectory):
    """
    Compare measurement with estimation and calculate error metrics and diagrams
    """
    df_matching = pd.read_csv(os.path.join(folder_matching, f"{scenario_name}_matching.csv"))
    measurement_prefix = df_matching['measurement_prefix'].values[0]
    df_trajectories = pd.read_csv(os.path.join(folder_trajectory, f'{scenario_name}_trajectories.csv'))
    df_measurement = pd.read_csv(os.path.join(folder_measurement, f"{measurement_prefix}INSSOL.csv"))

    df_measurement = prepare_measurement_data(df_measurement)

    relevant_track_id = df_matching['relevant_track_id'].values[0]

    start_timestamp = df_matching['start_timestamp'].values[0]
    datetimeformat = "%Y-%m-%d %H:%M:%S.%f"
    start_timestamp = datetime.datetime.strptime(start_timestamp, datetimeformat)

    df_trajectories = df_trajectories[df_trajectories['track_id'] == relevant_track_id]

    df_trajectories['timestamp'] = df_trajectories['frame_id'].apply(
        lambda x: get_timestamp_per_frame(x, start_timestamp),
        2)
    start_frame_id = df_trajectories['frame_id'].iloc[0]
    df_trajectories['t'] = df_trajectories['frame_id'].apply(
        lambda x: datetime.timedelta(seconds=(x - start_frame_id) / FPS_VIDEO).total_seconds(), 2)
    df_trajectories.set_index('timestamp', inplace=True)

    # df_trajectories = kalman_filter(df_trajectories)
    # df_trajectories = df_trajectories[(np.abs(stats.zscore(df_trajectories[['coordinate_world_x', 'coordinate_world_y']])) < 3).all(axis=1)]

    df_measurement.set_index('timestamp', inplace=True)

    # Join measurement and estimated trajectories
    result = df_measurement.join(df_trajectories, how='right')

    result.set_index('t', inplace=True)

    # calculate error metrics
    result['loss_x'] = result[['pos_x', 'coordinate_world_x']].apply(
        lambda x: np.absolute(x['pos_x'] - x['coordinate_world_x']), axis=1)
    result['loss_y'] = result[['pos_y', 'coordinate_world_y']].apply(
        lambda x: np.absolute(x['coordinate_world_y'] - x['pos_y']), axis=1)

    result['loss_x_opt'] = result[['pos_x', 'coordinate_world_x_opt']].apply(
        lambda x: np.absolute(x['pos_x'] - x['coordinate_world_x_opt']), axis=1)
    result['loss_y_opt'] = result[['pos_y', 'coordinate_world_y_opt']].apply(
        lambda x: np.absolute(x['coordinate_world_y_opt'] - x['pos_y']), axis=1)

    #  result['loss_x_opt_kf'] = result[['pos_x', 'coordinate_world_x_opt_kf']].apply(
    #    lambda x: np.absolute(x['pos_x'] - x['coordinate_world_x_opt_kf']), axis=1)
    #  result['loss_y_opt_kf'] = result[['pos_y', 'coordinate_world_y_opt_kf']].apply(
    #    lambda x: np.absolute(x['coordinate_world_y_opt_kf'] - x['pos_y']), axis=1)

    # calculate absolute error
    result['bbox_error'] = result[['loss_x', 'loss_y']].apply(lambda x: np.sqrt(x['loss_x'] ** 2 + x['loss_y'] ** 2),
                                                             axis=1)
    result['segm_error'] = result[['loss_x_opt', 'loss_y_opt']].apply(
        lambda x: np.sqrt(x['loss_x_opt'] ** 2 + x['loss_y_opt'] ** 2),
        axis=1)

    # rolling average for smoothing the diagram and makes it more readable
    result[['segm_error', 'bbox_error', 'loss_x', 'loss_y', 'loss_x_opt', 'loss_y_opt']] = \
        result[['segm_error', 'bbox_error', 'loss_x', 'loss_y', 'loss_x_opt', 'loss_y_opt']].rolling(7).mean()

    result = result.rename(columns={'pos_x': 'gt_x', 'pos_y': 'gt_y', 'coordinate_world_x': 'estimation_x',
                                    'coordinate_world_y': 'estimation_y', 'coordinate_world_x_opt': 'estimation_x_opt',
                                    'coordinate_world_y_opt': 'estimation_y_opt'})

    # result['full_loss_opt_kf'] = result[['loss_x_opt_kf', 'loss_y_opt_kf']].apply(lambda x: np.sqrt(x['loss_x_opt_kf'] ** 2 + x['loss_y_opt_kf'] ** 2),
    #                                                         axis=1)

    figsize = (6.0, 3.5)
    real_pos_x, ax_pos_x = plt.subplots(figsize=figsize)
    real_pos_y, ax_pos_y = plt.subplots(figsize=figsize)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    result[['gt_x', 'estimation_x', 'estimation_x_opt', 'loss_x_opt', 'loss_x']].plot(ax=ax_pos_x,
                                                                                      secondary_y=['loss_x',
                                                                                                   'loss_x_opt'],
                                                                                      color=['green', 'blue', 'red',
                                                                                             colors[4], colors[5]])
    result[['gt_y', 'estimation_y', 'estimation_y_opt', 'loss_y_opt', 'loss_y']].plot(ax=ax_pos_y,
                                                                                      secondary_y=['loss_y',
                                                                                                   'loss_y_opt'],
                                                                                      color=['green', 'blue', 'red',
                                                                                             colors[4], colors[5]])

    # Increase font size of matplotlib globally
    plt.rcParams.update({'font.size': 14})

    result[['velocity', 'gt_velocity']].plot()

    ax_pos_x.set_xlabel('Zeit [s]')
    ax_pos_x.set_ylabel('Position x (Longitude) [m]')
    ax_pos_x.right_ax.set_ylabel("Abweichung (loss) [m]")

    ax_pos_y.set_xlabel('Zeit [s]')
    ax_pos_y.set_ylabel('Position y (Latitude) [m]')
    ax_pos_y.right_ax.set_ylabel("Abweichung (loss) [m]")

    print(f"Current scenario: {scenario_name}")
    print("full_loss_opt", result['segm_error'].mean())
    print("full_loss", result['bbox_error'].mean())

    full_loss, ax = plt.subplots(figsize=figsize)

    result[['bbox_error', 'segm_error']].plot(ax=ax, use_index=True, color=['blue', 'red'])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [m]')
    #ax.legend(loc='upper right')

    if not is_plot_vis:
        plt.show()

    os.makedirs("plots_output", exist_ok=True)
    full_loss.savefig(os.path.join("plots_output", f"{scenario_name}_full_loss.png"), transparent=True,
                      bbox_inches='tight',
                      pad_inches=0.02)
    real_pos_x.savefig(os.path.join("plots_output", f"{scenario_name}_x_loss.png"), transparent=True,
                       bbox_inches='tight',
                       pad_inches=0.02)
    real_pos_y.savefig(os.path.join("plots_output", f"{scenario_name}_y_loss.png"), transparent=True,
                       bbox_inches='tight',
                       pad_inches=0.02)

    return result


def run_all_measurement_videos(szenarios):
    """
    Extract the trajectories on all measurement videos
    """
    for szenario in szenarios:
        print(f'Start traffic monitoring for {szenario}')
        process_video(f'../data/measurements/videos/{szenario}.mp4')


scenarios = []
for csv_file in os.listdir('../data/measurements/matching'):
    scenario = csv_file.split('_matching.csv')[0]
    scenarios.append(scenario)


# run_all_measurement_videos(scenarios)

def parse_args(default_scenario):
    parser = argparse.ArgumentParser('Evaluate trajectories estimation with measurement')
    parser.add_argument('--scenario', default=default_scenario, type=str,
                        help="specify scenario name for example marktkauf_04_audi_rechtsabbiegen")
    parser.add_argument('--all_scenarios', default=False, help='evaluate all scenarios in data folder',
                        action='store_true')

    parser.add_argument('--no_plot', default=False, help='avoid plotting during runtime',
                        action='store_true')

    parser.add_argument('--no_map_plot', default=False, help='visualize the matplotlib diagrams directly',
                        action='store_true')

    parser.add_argument('--data_folder_measurements', default=os.path.join('..', 'data', 'measurements', 'car_data'),
                        help="csv measurement file from car")
    parser.add_argument('--data_folder_trajectories_estimation',
                        default=os.path.join('..', 'data', 'measurements', 'videos', 'trajectory_output'))
    parser.add_argument('--data_folder_matching', default=os.path.join('..', 'data', 'measurements', 'matching'),
                        help="matching configuration for selecting track id and set manual timestamp")

    return parser.parse_args()


if __name__ == '__main__':
    # List of all scenarios for manual selection

    # scenario = "marktkauf_04_audi_rechtsabbiegen"
    # scenario = "marktkauf_05_klinikum_geradeaus"
    # scenario = "audi_08_klinikum_rechtsabbiegen"
    # scenario = "audi_09_marktkauf_linksabbiegen"
    # scenario = "audi_10_marktkauf_linksabbiegen"
    # scenario = "klinikum_01_marktkauf_geradeaus"
    # scenario = "audi_10_marktkauf_linksabbiegen"

    # scenario = "klinikum_01_stadtmitte_rechtsabbiegen"
    # scenario = "klinikum_01_marktkauf_geradeaus"
    # scenario = "stadtmitte_12_audi_geradeaus_02"
    # scenario = "audi_08_stadtmitte_geradeaus"
    # scenario = "marktkauf_07_stadtmitte_linksabbiegen"
    # scenario = "klinikum_01_stadtmitte_rechtsabbiegen"
    # scenario = "stadtmitte_11_marktkauf_rechtsabbiegen_02"

    args = parse_args(scenario)

    if args.all_scenarios:
        df_list = []
        for scenario in scenarios:
            print('Evaluate Szenario ' + scenario)
            joined_df = evaluate_one_scenario(scenario, args.no_plot, args.data_folder_measurements,
                                              args.data_folder_matching,
                                              args.data_folder_trajectories_estimation)
            df_list.append(joined_df)
            if not args.no_map_plot:
                show_measurement_in_map(joined_df, scenario)

        all_result = pd.concat(df_list)

        show_measurement_in_map(all_result, scenario)

        # print("RMS full_loss_opt", mean_squared_error(all_result['gt_x']))
        print("----- Loss over all measurement drives ------")
        print("RMS full_loss_opt", np.sqrt(np.mean((all_result['segm_error'] ** 2))))
        print("RMS full_loss", np.sqrt(np.mean((all_result['bbox_error'] ** 2))))
        print("full_loss_opt", all_result['segm_error'].mean())
        print("full_loss", all_result['bbox_error'].mean())
    else:
        joined_df = evaluate_one_scenario(args.scenario, args.no_plot, args.data_folder_measurements,
                                          args.data_folder_matching,
                                          args.data_folder_trajectories_estimation)

        if not args.no_map_plot:
            show_measurement_in_map(joined_df, scenario)
