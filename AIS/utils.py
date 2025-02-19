import pandas as pd
import pygeohash as pgh
import numpy as np
from geopy.distance import geodesic
from fastdtw import fastdtw
from typing import List

vectorized_geohash = np.vectorize(pgh.encode)

def v_geohash(lat: np.ndarray, lon: np.ndarray, precision: int) -> np.ndarray:
    """
    Vectorized geohash function
    :param lat: latitude
    :param lon: longitude
    :param precision: geohash precision
    :return: geohash
    """
    return vectorized_geohash(lat, lon, precision)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the haversine distance between two points
    :param lat1: latitude of the first point
    :param lon1: longitude of the first point
    :param lat2: latitude of the second point
    :param lon2: longitude of the second point
    :return: haversine distance
    """
    return geodesic((lat1, lon1), (lat2, lon2)).m

def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the dynamic time warping distance between two time series
    :param x: time series 1
    :param y: time series 2
    :return: dtw distance
    """
    distance, _ = fastdtw(x, y, dist=haversine)
    return distance


def feature_traj(ais_datas :List[pd.DataFrame], cols=None) -> pd.DataFrame:
    """
    计算特征轨迹
    :param cols: 经纬度的列名
    :param ais_datas: ais数据的列表
    :return: 计算的特征轨迹
    """
    if cols is None:
        cols = ["lat", "lon"]
    dis_matrix = np.zeros((len(ais_datas), len(ais_datas)))
    for i in range(len(ais_datas)):
        for j in range(i, len(ais_datas)):
            dis_matrix[i, j] = dis_matrix[j, i] = dtw_distance(ais_datas[i][cols].values, ais_datas[j][cols].values)
    return ais_datas[np.argmin(np.sum(dis_matrix, axis=1))]
