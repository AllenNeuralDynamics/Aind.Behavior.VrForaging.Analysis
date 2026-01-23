import csv
import io
import json
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, List, Optional
import pandas as pd
import requests
from dotmap import DotMap
from harp.reader import (DeviceReader, _ReaderParams, _create_register_handler,
                         create_reader, read_schema)


# Data stream sources
class Streams(DotMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, _dynamic=False)

    def list_streams(self) -> List[str]:
        return list(self.keys())

    def __str__(self):
        single_streams = [f"{key}: {value}" for key, value in self.items()]
        return f"Streams with {len(self)} streams: \n" + "\n".join(single_streams)

    def __repr__(self):
        return super().__repr__()


class DataStreamSource:
    """Represents a datastream source, usually comprised of various files from a single folder.
    These folders usually result from a single data acquisition logger"""

    def __init__(
        self,
        path: str | PathLike,
        name: Optional[str] = None,
        file_pattern_matching: str = "*",
        autoload=True,
    ) -> None:

        path = Path(path)
        self._path = path
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        self._name = name if name is not None else path.name
        self._files = [f for f in self._path.glob(file_pattern_matching)]
        self.populate_streams(autoload)

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        if self._path is None:
            raise ValueError("Path is not defined")
        if isinstance(self._path, str):
            self._path = Path(self._path)
        return self._path

    @property
    def files(self) -> List[Path]:
        return self._files

    def populate_streams(self, autoload) -> None:
        """Populates the streams attribute with a list of DataStream objects"""
        streams = [DataStream(file) for file in self.files]
        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = Streams({stream.name: stream for stream in streams})

    def __str__(self) -> str:
        return f"DataStreamSource from {self._path}"

    def __repr__(self) -> str:
        return f"DataStreamSource from {self._path}"


class SoftwareEventSource(DataStreamSource):
    def __init__(
        self,
        path: str | PathLike,
        name: str | None = None,
        file_pattern_matching: str = "*.json",
        autoload=True,
    ) -> None:
        super().__init__(path, name, file_pattern_matching, autoload=autoload)

    def populate_streams(self, autoload: bool) -> None:
        streams = [SoftwareEvent(file) for file in self.files]
        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = Streams({stream.name: stream for stream in streams})


def reader_from_url(
    device_yml_url: str, base_path: Optional[PathLike] = Path(".")
) -> DeviceReader:
    """Reads a device from a URL"""
    """Example: https://raw.githubusercontent.com/harp-tech/device.behavior/main/device.yml"""
    response = requests.get(device_yml_url)
    response.raise_for_status()
    if response.status_code == 200:
        _yml_stream = io.TextIOWrapper(io.BytesIO(response.content))
    else:
        raise ValueError(f"Failed to fetch device yml from {device_yml_url}")

    device = read_schema(_yml_stream, True)
    reg_readers = {
        name: _create_register_handler(
            device,
            name,
            _ReaderParams(base_path=base_path),
        )
        for name in device.registers.keys()
    }
    return DeviceReader(device, reg_readers)

def reader_from_yml(
    device_yml_path: PathLike, base_path: Optional[PathLike] = Path(".")
) -> DeviceReader:
    """
    Reads a device from a local YAML file (Windows path compatible).
    
    Example: Z:\\stage\\vr-foraging\\data\\828423\\828423_2025-11-17T224540Z\\behavior\\Behavior.harp\\device.yml
    """
    device_yml_path = Path(device_yml_path)
    if not device_yml_path.exists():
        raise FileNotFoundError(f"Device YAML file not found at {device_yml_path}")

    # Open file in text mode
    with device_yml_path.open("r", encoding="utf-8") as f:
        device = read_schema(f, True)

    reg_readers = {
        name: _create_register_handler(
            device,
            name,
            _ReaderParams(base_path=base_path),
        )
        for name in device.registers.keys()
    }

    return DeviceReader(device, reg_readers)

class HarpSource(DataStreamSource):
    def __init__(
        self,
        device: DeviceReader | Dict,
        path: str | PathLike,
        name: str | None = None,
        file_pattern_matching: str = "*",
        autoload=False,
        remove_suffix: Optional[str] = None,
    ) -> None:
        if isinstance(device, Dict):
            device = create_reader(**device)
            self._device = device
        elif isinstance(device, DeviceReader):
            self._device = device
        else:
            raise ValueError("device must be a HarpDevice or a string")
        self.remove_suffix = remove_suffix
        super().__init__(path, name, file_pattern_matching, autoload=autoload)

    @property
    def device(self) -> DeviceReader:
        return self._device

    def populate_streams(self, autoload: bool) -> None:
        if self.remove_suffix:
            streams = [
                HarpStream(
                    self.device, file, name=file.stem.replace(self.remove_suffix, "")
                )
                for file in self.files
            ]
        else:
            _inverted_device = self._invert_address_reg_mapping(self.device)
            streams = []
            for file in self.files:
                try:
                    streams.append(
                        HarpStream(
                            self.device,
                            file,
                            name=_inverted_device[int(file.stem.split("_")[-1])],
                        )
                    )
                except ValueError:
                    streams.append(
                        HarpStream(
                            self.device,
                            file,
                            name=file.stem.split("_")[-1])
                        )
                except KeyError:
                    Warning(f"Could not find a register for {file}")
                except ValueError as e:
                    if file.stem.split("_")[-1] == "device":
                        pass
                    else:
                        raise e
        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = Streams({stream.name: stream for stream in streams})

    @staticmethod
    def _invert_address_reg_mapping(device: DeviceReader) -> Dict[int, str]:
        return {v.address: k for k, v in device.device.registers.items()}


class ConfigSource(DataStreamSource):
    def __init__(
        self,
        path: str | PathLike,
        name: str | None = None,
        file_pattern_matching: str = "*.json",
        autoload=True,
    ) -> None:
        super().__init__(path, name, file_pattern_matching, autoload=autoload)

    def populate_streams(self, autoload: bool) -> None:
        streams = [Config(file) for file in self.files]
        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = Streams({stream.name: stream for stream in streams})


class OperationControlSource(DataStreamSource):
    def __init__(
        self,
        path: str | PathLike,
        name: str | None = None,
        file_pattern_matching: str = "*.csv",
        autoload=True,
    ) -> None:
        super().__init__(path, name, file_pattern_matching, autoload=autoload)

    def populate_streams(self, autoload: bool) -> None:
        streams: List[DataStream] = []
        for file in self.files:
            streams.append(
                DataStream(path=file, data_type=DataStreamType.CSV, reader=self._loader)
            )

        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = Streams({stream.name: stream for stream in streams})

    @staticmethod
    def _loader(path: Path | str):
        _exists = False
        with open(path) as csvfile:
            try:
                has_header = csv.Sniffer().has_header(csvfile.read(20_480))
                _exists = True
            except csv.Error:
                Warning(f"Could not determine if {path} has a header")
                has_header = False
        if _exists is False:
            df = pd.DataFrame()
            df.index.names = ["Seconds"]
            return df
        if has_header:
            df = pd.read_csv(path, header=0, index_col=0)
        else:
            df = pd.read_csv(path, header=None, index_col=0)
        return df


## Data stream types


class DataStreamType(Enum):
    """Represents the available DataStream types"""

    NULL = 0
    CUSTOM = 1
    HARP = 2
    JSON = 3
    SOFTWARE_EVENT = 4
    VIDEO = 5
    CSV = 6


class DataStream:
    """Represents a single datastream file"""

    def __init__(
        self,
        path: Optional[str | PathLike] = None,
        name: Optional[str] = None,
        data_type: DataStreamType = DataStreamType.NULL,
        reader: Optional[Callable] = None,
        parser: Optional[Callable] = None,
    ) -> None:
        if path:
            path = Path(path)
            self._path = path
            if not path.is_file():
                raise ValueError(f"Path {path} is not a file")
            self._name = name if name is not None else path.stem
        else:
            if name is None:
                raise ValueError("Either path or name must be provided")
        self._dataType = data_type
        self.reader = reader
        self.parser = parser
        self._data = None

    @property
    def data(self) -> any:
        if self._data is None:
            raise ValueError(
                "Data is not loaded. \
                             Try self.data(populate=True) to attempt\
                             to automatically load it"
            )
        return self._data

    @property
    def data_type(self) -> DataStreamType:
        return self._dataType

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        if self._path is None:
            raise ValueError("Path is not defined")
        if isinstance(self._path, str):
            self._path = Path(self._path)
        return self._path

    def load_from_file(
        self,
        reader: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        """Loads the data stream from a file into memory"""
        force_reload = True if self._data is None else force_reload
        if force_reload:
            reader = reader if reader is not None else self.reader
            if reader:
                self._data = reader(self._path)
                return self._data
            else:
                raise NotImplementedError(
                    "A valid .load_from_file() method must be implemented,\
                        or a reader function must be provided"
                )

    @classmethod
    def parse(self, value: any, **kwargs):
        """Loads the data stream from a value"""
        ds = DataStream(kwargs)
        if ds.parser:
            ds._data = ds.parser(value)
            return ds
        else:
            raise NotImplementedError(
                "A valid .parse() method must be implemented,\
                    or a parse function must be provided"
            )

    def __str__(self) -> str:
        if self._data is not None:
            return f"{self._dataType} stream with {len(self._data)} entries"
        else:
            return f"{self._dataType} stream with None/Not loaded entries"

    def __repr__(self) -> str:
        if self._data is not None:
            return f"{self._dataType} stream with {len(self._data)} entries"
        else:
            return f"{self._dataType} stream with None/Not loaded entries"


class HarpStream(DataStream):
    def __init__(self, device: DeviceReader, path: Optional[Path] = None, **kwargs):
        if isinstance(device, DeviceReader):
            self._device = device
        else:
            raise ValueError("device must be a DeviceReader")
        super().__init__(
            path=path,
            **kwargs,
            data_type=DataStreamType.HARP,
            reader=None,
            parser=None,
        )

    @property
    def device(self) -> DeviceReader:
        return self._device

    def load_from_file(
        self, path: Optional[Path] = None, force_reload: bool = False
    ) -> None:
        """Loads the datastream from a file"""
        force_reload = True if self._data is None else force_reload
        if force_reload:
            if path is None:
                path = self._path
            # load raw file as a binary
            reg_addr = self._get_address_from_bin(path)
            self._data = self.device.registers[reg_addr].read(path, keep_type=True)

    @staticmethod
    def _get_address_from_bin(path: PathLike) -> int:
        with open(path, "rb") as file:
            file.seek(2)
            address = file.read(1)[0]
        return address


class SoftwareEvent(DataStream):
    """Represents a generic Software event."""

    def __init__(self, path: Optional[str | PathLike] = None, **kwargs):
        super().__init__(
            path=path,
            **kwargs,
            data_type=DataStreamType.JSON,
            reader=None,
            parser=None,
        )

    def _load_single_event(self, value: str) -> None:
        return json.loads(value)

    def load_from_file(
        self, path: Optional[str | PathLike] = None, force_reload: bool = False
    ) -> None:
        """Loads the datastream from a file"""
        force_reload = True if self._data is None else force_reload
        if force_reload:
            if path is None:
                path = self._path
            with open(path, "r") as f:
                self._data = pd.DataFrame(
                    [self._load_single_event(value=event) for event in f.readlines()]
                )
                self._data.rename(columns={"timestamp": "Seconds"}, inplace=True)
                self._data.set_index("Seconds", inplace=True)

    def json_normalize(self, *args, **kwargs):
        df = pd.concat(
            [
                self.data,
                pd.json_normalize(self._data["data"], args, kwargs).set_index(
                    self.data.index
                ),
            ],
            axis=1,
        )
        return df

    @classmethod
    def parse(self, value: str, **kwargs) -> pd.DataFrame:
        """Loads the datastream from a value"""
        ds = SoftwareEvent(**kwargs)
        ds._data = pd.DataFrame(
            [SoftwareEvent._load_single_event(value=line) for line in value.split("\n")]
        )
        ds._data.rename(columns={"timestamp": "Seconds"}, inplace=True)
        ds._data.set_index("Seconds", inplace=True)
        return ds


class Config(DataStream):
    """Represents a generic Software event."""

    def __init__(self, path: Optional[str | PathLike] = None, **kwargs):
        super().__init__(
            path=path,
            **kwargs,
            data_type=DataStreamType.JSON,
            reader=None,
            parser=None,
        )

    def load_from_file(
        self, path: Optional[str | PathLike] = None, force_reload: bool = False
    ) -> None:
        """Loads the datastream from a file"""
        force_reload = True if self._data is None else force_reload
        if force_reload:
            if path is None:
                path = self._path
            with open(path, "r") as f:
                self._data = json.load(f)

    @classmethod
    def parse(self, value: str, **kwargs) -> Dict:
        """Loads the datastream from a value"""
        ds = Config(**kwargs)
        ds._data = json.load(value)
        return ds
