import numpy as np
import pandas as pd
import clr

_SECONDS_PER_TICK = 32e-6
_payloadtypes = {
                1 : np.dtype(np.uint8),
                2 : np.dtype(np.uint16),
                4 : np.dtype(np.uint32),
                8 : np.dtype(np.uint64),
                129 : np.dtype(np.int8),
                130 : np.dtype(np.int16),
                132 : np.dtype(np.int32),
                136 : np.dtype(np.int64),
                68 : np.dtype(np.float32)
                }


def read_harp_bin(file):

    data = np.fromfile(file, dtype=np.uint8)

    if len(data) == 0:
        return None

    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = _payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    ticks = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = ticks * _SECONDS_PER_TICK + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data, offset=11,
        strides=(stride, elementsize))

    if payload.shape[1] ==  1:
        ret_pd = pd.DataFrame(payload, index=seconds, columns= ["Value"])
        ret_pd.index.names = ['Seconds']

    else:
        ret_pd =  pd.DataFrame(payload, index=seconds)
        ret_pd.index.names = ['Seconds']

    return ret_pd


def read_bytes_from_bin(filename):
    binary_data = np.fromfile(filename, dtype=np.uint8)
    msg_len = binary_data[1] + 2
    n_msg = len(binary_data) // msg_len
    data_array = binary_data.reshape(n_msg, msg_len)
    return data_array


_CORE_IMPORTED = False


def import_core():
    clr.AddReference("Bonsai.Harp")
    globals()["Bonsai"] = __import__("Bonsai", fromlist=["Harp"])
    globals()["Bonsai.Harp"] = getattr(globals()["Bonsai"], "Harp")
    globals()["System"] = __import__("System")
    global _CORE_IMPORTED
    _CORE_IMPORTED = True


class HarpInterface:
    if _CORE_IMPORTED is False:
        import_core()

    def __init__(self, Device) -> None:
        self.RegisterMap = Device.RegisterMap
        self.Device = Device

    def message_to_payload_method(self, *arg, **kwargs) -> any:
        raise NotImplementedError("Must be implemented!")

    def harp_messages_to_payloads(self,
                                  data: np.array,
                                  address=None
                                  ) -> np.array:
        get_payload_method = self.message_to_payload_method(
            data[0].Address if address is None else address)
        vectorized_method = np.vectorize(get_payload_method)
        return vectorized_method(data)

    def file_to_payloads(self, filename: str) -> np.array:
        data = self.read_harp_bin_to_messages(filename)
        return self.harp_messages_to_payloads(data)

    def file_to_dataframe(self, filename: str) -> pd.DataFrame:
        payloads = self.file_to_payloads(filename)
        df = pd.DataFrame([[entry.Seconds, entry.Value] for entry in payloads],
                          columns=["Seconds", "Value"])
        df.set_index("Seconds", inplace=True)
        return df

    @staticmethod
    def bytes_to_harp_message(in_bytes) -> Bonsai.Harp.HarpMessage:
        return Bonsai.Harp.HarpMessage(in_bytes)

    def read_harp_bin_to_messages(self, filename: str) -> np.array:
        return self.array_to_harp_messages(read_bytes_from_bin(filename))

    def array_to_harp_messages(self, data: np.array) -> np.array:
        return np.apply_along_axis(
            lambda x: self.bytes_to_harp_message(x.tobytes()
                                                 ), axis=1, arr=data)

    @staticmethod
    def read_bytes_from_bin(filename):
        read_bytes_from_bin(filename)

    @staticmethod
    def unpack_enum_flag(df: pd.DataFrame,
                         flag_class: object,
                         exclude_default_flag: bool = True) -> pd.DataFrame:
        if exclude_default_flag:
            flags = [val.ToString() for val in System.Enum.GetValues(flag_class) if int(val) > 0]
        else:
            flags = [val.ToString() for val in System.Enum.GetValues(flag_class)]

        for flag in flags:
            df[flag] = df["Value"].apply(lambda entry: entry.HasFlag(System.Enum.Parse(flag_class, flag)))
        return df


class HarpDevice(HarpInterface):
    def __init__(self, board_name: str):
        clr.AddReference(f"Harp.{board_name}")
        package = __import__('Harp', fromlist=[board_name])
        module = getattr(package, board_name)
        self.module = module
        super().__init__(module.Device)

    def message_to_payload_method(self, msg_or_address: Bonsai.Harp.HarpMessage | int) -> any:
        if isinstance(msg_or_address, int):
            address = msg_or_address
        elif isinstance(msg_or_address, Bonsai.Harp.HarpMessage):
            address = msg_or_address.Address
        else:
            raise ValueError("msg_or_address must be either a HarpMessage or an int")

        try:
            _type = self.Device.RegisterMap[address]
        except KeyError:
            raise ValueError(f"Address {address} not found in RegisterMap")

        GetPayloadMethod = _type.GetMethod("GetTimestampedPayload")
        return lambda msg: GetPayloadMethod.Invoke(None, [msg])

