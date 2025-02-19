import numpy as np
import pandas as pd
from typing import Tuple
from geopy.distance import geodesic

from MarAIS.AIS.utils import feature_traj


def noise_detection(ais_data: pd.DataFrame, v_max: float, delta_cog:float, cols=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects noise in the AIS data
    :param ais_data: AIS 数据，包含经纬度、速度、航向、时间等信息，DataFrame格式
    :param v_max: 最大速度，单位 m/s
    :param cols:包含经纬度、速度、航向、时间等信息的列名
    :param delta_cog: 航向变化的最大值
    :return: 返回异常值的索引
    """
    if cols is None:
        cols = ["lat", "lon", "SOG", "COG", "time"]

    Lat = ais_data[cols[0]].values
    Lon = ais_data[cols[1]].values
    SOG = ais_data[cols[2]].values
    COG = ais_data[cols[3]].values

    # Calculate the distance between two consecutive points
    d = np.array([geodesic((Lat[i], Lon[i]), (Lat[i + 1], Lon[i + 1])).m for i in range(len(Lat) - 1)])
    # 计算相邻两点时间间隔，单位为秒
    dt = np.diff(ais_data[cols[4]].values).astype(float) / 1e9
    # 利用距离和时间差来近似计算速度
    v = d / dt
    noise_position = np.unique(np.concatenate([np.where(v > v_max)[0], np.where(v > v_max)[0] + 1]))

    # 计算相邻两点速度的均值
    v_mean = (SOG[:-1] + SOG[1:]) / 2
    v_mean = np.insert(v_mean, 0, SOG[0])
    # 按照规则选取速度的异常值
    noise_speed1 = np.where(SOG > v_max)[0]
    noise_speed2 = np.where(SOG > 1.2 * v_mean)[0]
    noise_speed3 = np.where(SOG < 0.8 * v_mean)[0]
    noise_speed = np.unique(np.concatenate([noise_speed1, noise_speed2, noise_speed3]))
    # 按照规则选取航向的异常值
    cog_diff = np.abs(np.diff(COG))
    cog_diff = np.array([min(np.abs(cog_diff[i]), 360 - np.abs(cog_diff[i])) for i in range(len(cog_diff))])
    noise_cog = np.where(cog_diff > delta_cog)[0]
    noise_cog = np.unique(np.concatenate([noise_cog, noise_cog + 1]))

    # 返回异常值的索引
    return noise_position, noise_speed, noise_cog

def missing_detection(ais_data: pd.DataFrame, average_time_gap: float, n_threshold :int, col=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects missing data in the AIS data
    :param ais_data: AIS 数据，包含经纬度、速度、航向、时间等信息，DataFrame格式
    :param average_time_gap: 平均时间间隔，单位为秒
    :param n_threshold: 缺失数据数量的阈值
    :param col: 时间列的列名
    :return: 返回缺失数据的索引
    """
    if col is None:
        col = "time"

    # 计算相邻两点时间间隔，单位为秒
    dt = np.diff(ais_data[col]).astype(float) / 1e9
    missing_time_a = np.where(dt > average_time_gap * n_threshold)[0]
    missing_time_b = missing_time_a + 1
    return missing_time_a, missing_time_b

def reconstruct_sog_position(ais_data: pd.DataFrame, noise_position: np.ndarray, noise_speed: np.ndarray, cols=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs the AIS data by removing noise and filling missing data
    :param ais_data: AIS 数据，包含经纬度、速度、航向、时间等信息，DataFrame格式
    :param noise_position: 位置异常值的索引
    :param noise_speed: 速度异常值的索引
    :param cols: 包含经纬度、速度、航向、时间等信息的列名
    :return: 返回处理后的AIS数据
    """
    if cols is None:
        cols = ["lat", "lon", "SOG", "COG", "time"]

    t = ais_data[cols[4]].astype(float) / 1e9
    t = (t - t[0]).values   # 时间戳

    Lon = ais_data[cols[1]].values
    Lat = ais_data[cols[0]].values
    SOG = ais_data[cols[2]].values

    # Interpolate the missing data
    lon_inter = interpolate_sog_position(t, Lon, noise_position)
    lat_inter = interpolate_sog_position(t, Lat, noise_position)
    sog_inter = interpolate_sog_position(t, SOG, noise_speed)

    return lon_inter, lat_inter, sog_inter

def interpolate_sog_position(time: np.ndarray, values: np.ndarray, noise_index: np.ndarray) -> np.ndarray:
    """
    Interpolates the missing data in the AIS data
    :param time: 时间戳
    :param values: 待插值的数据
    :param noise_index: 缺失数据的索引
    :return: 返回插值后的数据
    """
    v = np.vander(time, increasing=True)
    a = np.linalg.solve(v, values)
    inter_values = np.dot(v[noise_index, :], a)
    values[noise_index] = inter_values

    return values

def reconstruct_cog(ais_data: pd.DataFrame, noise_cog: np.ndarray, cols=None) -> np.ndarray:
    """
    Reconstructs the AIS data by removing noise and filling missing data
    :param ais_data: AIS 数据，包含经纬度、速度、航向、时间等信息，DataFrame格式
    :param noise_cog: 航向异常值的索引
    :param cols: 包含经纬度、速度、航向、时间等信息的列名
    :return: 返回处理后的AIS数据
    """
    if cols is None:
        cols = ["lat", "lon", "SOG", "COG", "time"]

    t = ais_data[cols[4]].astype(float) / 1e9
    dt1 = np.array(t)
    dt2 = t[2:] - t[:-2]

    COG = ais_data[cols[3]].values

    rot = COG[2:] - COG[:-2]
    rot = np.array([rot if rot < 180 else 360 - rot for rot in rot])
    rot = rot / dt2

    c = rot * (dt1[:-1]) + COG[:-2]
    c = np.array([c if c < 360 else c - 360 for c in c])

    # 非异常值的索引
    normal_index = np.setdiff1d(np.arange(len(COG)), noise_cog)
    # 0和最后一个位置的插值变为距离最近的非异常值
    c = np.insert(c, 0, COG[normal_index[0]])
    c = np.insert(c, len(c), COG[normal_index[-1]])

    COG[noise_cog] = c[noise_cog]

    return COG

def reconstruct_noise(ais_data: pd.DataFrame, v_max: float, delta_cog: float, cols=None) -> pd.DataFrame:
    """
    Reconstructs the AIS data by removing noise and filling missing data
    :param ais_data: AIS 数据，包含经纬度、速度、航向、时间等信息，DataFrame格式
    :param v_max: 最大速度，单位 m/s
    :param delta_cog: 航向变化的最大值
    :param cols: 包含经纬度、速度、航向、时间等信息的列名
    :return: 返回处理后的AIS数据
    """
    if cols is None:
        cols = ["lat", "lon", "SOG", "COG", "time"]

    noise_position, noise_speed, noise_cog = noise_detection(ais_data, v_max, delta_cog, cols)

    lon_inter, lat_inter, sog_inter = reconstruct_sog_position(ais_data, noise_position, noise_speed, cols)
    cog_inter = reconstruct_cog(ais_data, noise_cog, cols)

    ais_data[cols[0]] = lat_inter
    ais_data[cols[1]] = lon_inter
    ais_data[cols[2]] = sog_inter
    ais_data[cols[3]] = cog_inter

    return ais_data


def reconstruct_missing(ais_data: pd.DataFrame, average_time_gap: float, n_threshold: int, feature_ais: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Reconstructs the AIS data by removing noise and filling missing data
    :param ais_data: AIS 数据，包含经纬度、速度、航向、时间等信息，DataFrame格式
    :param average_time_gap: 平均时间间隔，单位为秒
    :param n_threshold: 缺失数据数量的阈值
    :param feature_ais: 特征轨迹
    :param cols: 包含经纬度、速度、航向、时间等信息的列名
    :return: 返回处理后的AIS数据
    """
    if cols is None:
        cols = ["lat", "lon", "SOG", "COG", "time"]

    missing_time_a, missing_time_b = missing_detection(ais_data, average_time_gap, n_threshold, cols[4])

    missing_pd = []

    for i in range(len(missing_time_a)):
        lat1, lat2 = ais_data[cols[0]].iloc[missing_time_a[i]], ais_data[cols[0]].iloc[missing_time_b[i]]
        lon1, lon2 = ais_data[cols[1]].iloc[missing_time_a[i]], ais_data[cols[1]].iloc[missing_time_b[i]]
        dist1 = np.array([geodesic((lat1, lon1), (feature_ais.loc[i, cols[0]],feature_ais.loc[i, cols[1]])).m for i in range(len(feature_ais))])
        index1 = np.argmin(dist1)
        dist2 = np.array([geodesic((lat2, lon2), (feature_ais.loc[i, cols[0]],feature_ais.loc[i, cols[1]])).m for i in range(len(feature_ais))])
        index2 = np.argmin(dist2)
        index_a = np.min([index1, index2])
        index_b = np.max([index1, index2])
        inter_x, inter_y, inter_soc, inter_cog, inter_time = inter_missing(ais_data.iloc[missing_time_a[i]], ais_data.iloc[missing_time_b[i]], feature_ais.iloc[index_a:index_b], cols)
        inter_pd = pd.DataFrame({cols[0]: inter_y, "lon": cols[1], cols[2]: inter_soc, cols[3]: inter_cog, cols[4]: inter_time})
        missing_pd.append(inter_pd)

    ais_data = pd.concat([ais_data, *missing_pd], ignore_index=True)
    ais_data = ais_data.sort_values(by=cols[4]).reset_index(drop=True)
    return ais_data

def inter_missing(point1: pd.Series, point2: pd.Series, feature_ais: pd.DataFrame, cols=None):
    """
    Interpolates the missing data in the AIS data
    :param point1: 缺失数据的第一个点
    :param point2: 缺失数据的第二个点
    :param feature_ais: 特征轨迹
    :param cols: 包含经纬度、速度、航向、时间等信息的列名
    :return: 返回插值后的数据
    """
    if cols is None:
        cols = ["lat", "lon", "SOG", "COG", "time"]
    delta_x = ((feature_ais.iloc[-1][cols[1]] - point2[cols[1]]) - (
                feature_ais.iloc[0][cols[1]] - point1[cols[1]])) / len(feature_ais)
    delta_y = ((feature_ais.iloc[-1][cols[0]] - point2[cols[0]]) - (
                feature_ais.iloc[0][cols[0]] - point1[cols[0]])) / len(feature_ais)
    delta_soc = ((feature_ais.iloc[-1][cols[2]] - point2[cols[2]]) - (
                feature_ais.iloc[0][cols[2]] - point1[cols[2]])) / len(feature_ais)
    delta_cog = ((feature_ais.iloc[-1][cols[3]] - point2[cols[3]]) - (
                feature_ais.iloc[0][cols[3]] - point1[cols[3]])) / len(feature_ais)
    delta_time = ((feature_ais.iloc[-1][cols[4]] - point2[cols[4]]) - (
                feature_ais.iloc[0][cols[4]] - point1[cols[4]])) / len(feature_ais)

    inter_x = feature_ais.loc[1:len(feature_ais)-2, cols[1]] + delta_x
    inter_y = feature_ais.loc[1:len(feature_ais)-2, cols[0]] + delta_y
    inter_soc = feature_ais.loc[1:len(feature_ais)-2, cols[2]] + delta_soc
    inter_cog = feature_ais.loc[1:len(feature_ais)-2, cols[3]] + delta_cog
    inter_time = feature_ais.loc[1:len(feature_ais)-2, cols[4]] + delta_time

    return inter_x, inter_y, inter_soc, inter_cog, inter_time   # 返回插值后的数据