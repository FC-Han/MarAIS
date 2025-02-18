"""
Module Name: GFW API Module

This module provides functions for GFW API.

Features:
-
"""

import os
from urllib.parse import urlparse, parse_qs
import time
import numpy as np
import requests
from typing import Union
from error import InvalidInputError


class GFW:
    """
    """

    def __init__(self, token: str):
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    def fishing_hour(self, start: str, end: str, region: object, filters: str = None, spatial_resolution: str = "low",
                     fm: str = "csv",
                     group_by: str = "gearType", temporal_resolution: str = "daily", spatial_aggregation: bool = True,
                     dataset: str = "public-global-fishing-effort:latest"):
        """
        :type region: object
        :param region:
        :param filters:
        :param start:
        :param end:
        :param spatial_resolution:
        :param fm:
        :param group_by:
        :param temporal_resolution:
        :param spatial_aggregation:
        :param dataset
        :return:
        """
        url = 'https://gateway.api.globalfishingwatch.org/v3/4wings/report'
        params = {
            'spatial-resolution': spatial_resolution,
            'temporal-resolution': temporal_resolution,
            "group-by": group_by,
            "format": fm,
            "datasets[0]": dataset,
            "spatial-aggregation": spatial_aggregation,
            "filters[0]": filters,
            "date-range": start + "," + end
        }
        response = requests.post(url, headers=self.headers, params=params, json=region)

        if response.status_code == 200:
            os.makedirs("fishing_time", exist_ok=True)
            zip_filepath = "fishing_time" + start + "-" + end + ".zip"
            # 将响应内容保存为ZIP文件
            with open(zip_filepath, 'wb') as zip_file:
                zip_file.write(response.content)

                print(f"Saved ZIP file: {zip_filepath}")
                time.sleep(2)
        else:
            print("skip " + start + "-" + end)

    @staticmethod
    def create_region_eez(eez_id: int) -> object:
        return {
            "region": {
                "dataset": "public-eez-areas",
                "id": eez_id
            }
        }

    @staticmethod
    def create_region_mpa(mpa_id: int) -> object:
        return {
            "region": {
                "dataset": "public-mpa-all",
                "id": mpa_id
            }
        }

    @staticmethod
    def create_region_geojson(coordinates):
        return {
            "geojson": {
                "type": "Polygon",
                "coordinates": coordinates
            }
        }

    def get_region_id(self, region_name: str, region_source: str) -> Union[int, str]:
        base_url = "https://gateway.api.globalfishingwatch.org/v3/datasets/"
        if region_source == "EEZ":
            base_url += "public-eez-areas/context-layers"
        elif region_source == "MPA":
            base_url += "public-mpa-all/context-layers"
        elif region_source == "RFMO":
            base_url += "public-rfmo/context-layers"
        else:
            raise InvalidInputError(region_source, ["EEZ", "MPA", "RFMO"])

        response = requests.get(base_url, headers=self.headers)
        if response.status_code == 200:
            # 如果响应成功，打印返回的 JSON 数据
            data = response.json()
        else:
            # 如果响应失败，打印错误信息
            raise Exception(f"Error {response.status_code}: {response.text}")

        if region_source == "EEZ":
            return [d["id"] for d in data if d["iso3"] == region_name][0]
        else:
            return [d["id"] for d in data if d["label"] == region_name][0]
