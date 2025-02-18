from typing import Union, List

# 定义支持的类型
SUPPORTED_TYPES = [
    "PositionReport", "AddressedSafetyMessage",
    "AddressedBinaryMessage", "AidsToNavigationReport", "AssignedModeCommand",
    "BaseStationReport", "BinaryAcknowledge",
    "ChannelManagement", "CoordinatedUTCInquiry", "DataLinkManagementMessage",
    "ExtendedClassBPositionReport",
    "GnssBroadcastBinaryMessage", "Interrogation",
    "LongRangeAisBroadcastMessage", "MultiSlotBinaryMessage",
    "SafetyBroadcastMessage", "ShipStaticData", "SingleSlotBinaryMessage",
    "StandardClassBPositionReport", "StandardSearchAndRescueAircraftReport",
    "StaticDataReport"
]

def validate_message_types(input_message_types: Union[str, List[str]]) -> None:
    """
    验证输入数据是否为支持的类型或支持类型的列表
    :param input_message_types: 输入的数据，可以是单个类型或类型列表
    """
    if isinstance(input_message_types, str):
        if input_message_types not in SUPPORTED_TYPES:
            raise ValueError(f"输入的 '{input_message_types}' 不是支持的类型。")
    elif isinstance(input_message_types, list):
        for item in input_message_types:
            if not isinstance(item, str) or item not in SUPPORTED_TYPES:
                raise ValueError(f"列表中的元素 '{item}' 不是支持的类型。")
    else:
        raise ValueError("输入必须是字符串或字符串列表。")