import asyncio
import websockets
import json
from typing import Union, List
from message_types import validate_message_types
import aiofiles


class InvalidInputError(Exception):
    pass


def create_bounding_box(min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    if min_lon < -180 or min_lon > 180:
        raise InvalidInputError("Longitude must be between -180 and 180")
    if max_lon < -180 or max_lon > 180:
        raise InvalidInputError("Longitude must be between -180 and 180")
    if min_lat < -90 or min_lat > 90:
        raise InvalidInputError("Latitude must be between -90 and 90")
    if max_lat < -90 or max_lat > 90:
        raise InvalidInputError("Latitude must be between -90 and 90")
    if min_lon > max_lon:
        raise InvalidInputError("min_lon must be less than max_lon")
    if min_lat > max_lat:
        raise InvalidInputError("min_lat must be less than max_lat")
    return [[min_lat, min_lon], [max_lat, max_lon]]

def create_bounding_boxs(min_lons: List[float], min_lats: List[float], max_lons: List[float], max_lats: List[float]):
    if len(min_lons) != len(min_lats) or len(min_lons) != len(max_lons) or len(min_lons) != len(max_lats):
        raise InvalidInputError("Length of min_lons, min_lats, max_lons, max_lats must be the same")
    return [[min_lat, min_lon, max_lat, max_lon] for min_lon, min_lat, max_lon, max_lat in zip(min_lons, min_lats, max_lons, max_lats)]

class Stream:
    def __init__(self, api_key):
        self.api_key = api_key

    async def connect_ais_stream(self, bounding_boxs: list, filter_mmsi: Union[str, List[str]] = None, filter_message_type: Union[str, List[str]] = None, output_folder: str = ""):
        async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
            subscribe_message = {
                "APIKey": self.api_key,
                "BoundingBoxes": bounding_boxs,
            }
            if isinstance(filter_mmsi, list) and len(filter_mmsi) > 50:
                raise InvalidInputError("Maximum number of MMSI filters is 50")

            if filter_mmsi:
                subscribe_message["FiltersShipMMSI"] = filter_mmsi if isinstance(filter_mmsi, list) else [filter_mmsi]
            if filter_message_type:
                validate_message_types(filter_message_type)
                subscribe_message["FilterMessageTypes"] = filter_message_type if isinstance(filter_message_type, list) else [filter_message_type]

            subscribe_message_json = json.dumps(subscribe_message)
            await websocket.send(subscribe_message_json)

            async for message in websocket:
                message_type = message["MessageType"]
                message_content = message["Message"][message_type]

                async with aiofiles.open(output_folder + f"{message_type}.json", "a") as f:
                    await f.write(json.dumps(message_content) + "\n")
