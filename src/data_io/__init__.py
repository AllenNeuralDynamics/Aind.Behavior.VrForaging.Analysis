import pandas as pd
import json
from dataclasses import dataclass
from typing import Optional, Callable, List
from pathlib import Path
from dotmap import DotMap
from enum import Enum
import harp
# Data stream sources


class DataStreamSource:
    """Represents a datastream source, usually comprised of various files from a single folder.
    These folders usually result from a single data acquisition logger"""
    def __init__(self,
                 path: str | Path,
                 name: Optional[str] = None,
                 file_pattern_matching: str = "*",
                 autoload=True,
                 ) -> None:

        if isinstance(path, str):
            path = Path(path)
        self._path = path
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        self._name = name if name is not None else path.name
        self.files = [f for f in self._path.glob(file_pattern_matching)]
        self.populate_streams(autoload)

    def populate_streams(self, autoload) -> DotMap:
        """Populates the streams attribute with a list of DataStream objects"""
        streams = [DataStream(file) for file in self.files]
        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = DotMap({stream.name: stream for stream in streams})

    def __str__(self) -> str:
        return f"DataStreamSource from {self._path}"

    def __repr__(self) -> str:
        return f"DataStreamSource from {self._path}"


class SoftwareEventSource(DataStreamSource):
    def __init__(self, path: str | Path,
                 name: str | None = None,
                 file_pattern_matching: str = "*.json",
                 autoload=True) -> None:
        super().__init__(path, name, file_pattern_matching, autoload=autoload)

    def populate_streams(self, autoload: bool) -> DotMap:
        streams = [SoftwareEvent(file) for file in self.files]
        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = DotMap({stream.name: stream for stream in streams})


class HarpSource(DataStreamSource):
    def __init__(self, device: harp.HarpDevice | str,
                 path: str | Path,
                 name: str | None = None,
                 file_pattern_matching: str = "*",
                 autoload=False,
                 remove_suffix: Optional[str] = "Register__") -> None:
        if isinstance(device, str):
            device = harp.HarpDevice(device)
            self.device = device
        elif isinstance(device, harp.HarpDevice):
            self.device = device
        else:
            raise ValueError("device must be a HarpDevice or a string")
        self.remove_suffix = remove_suffix
        super().__init__(path, name, file_pattern_matching, autoload=autoload)

    def populate_streams(self, autoload: bool) -> DotMap:
        if self.remove_suffix:
            streams = [HarpStream(
                self.device,
                file,
                name=file.stem.replace(self.remove_suffix, ""))
                for file in self.files]
        else:
            streams = [HarpStream(self.device, file) for file in self.files]
        if autoload is True:
            for stream in streams:
                stream.load_from_file()
        self.streams = DotMap({stream.name: stream for stream in streams})


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
    def __init__(self,
                 path: Optional[str | Path] = None,
                 name: Optional[str] = None,
                 data_type: DataStreamType = DataStreamType.NULL,
                 reader: Optional[Callable] = None,
                 parser: Optional[Callable] = None,
                 ) -> None:
        if path:
            if isinstance(path, str):
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
    def data(self,
             populate: bool = False,
             force_reload: bool = False,
             ) -> any:
        if populate is True:
            self.load_from_file(force_reload=force_reload)
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

    def load_from_file(self,
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
                        or a file_reader function must be provided")

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
                    or a reader function must be provided")

    def __str__(self) -> str:
        return f"{self._dataType} stream with {len(self._data)} entries"

    def __repr__(self) -> str:
        return f"{self._dataType} stream with {len(self._data)} entries"


class HarpStream(DataStream):
    def __init__(self,
                 device: harp.HarpDevice | str,
                 path: Optional[Path] = None, **kwargs):
        if isinstance(device, str):
            device = harp.HarpDevice(device)
            self.device = device
        elif isinstance(device, harp.HarpDevice):
            self.device = device
        else:
            raise ValueError("device must be a HarpDevice or a string")
        super().__init__(
            path=path, **kwargs,
            data_type=DataStreamType.HARP,
            reader=None,
            parser=None,
            )

    def load_from_file(self,
                       path: Optional[Path] = None,
                       force_reload: bool = False) -> None:
        """Loads the datastream from a file"""
        force_reload = True if self._data is None else force_reload
        if force_reload:
            if path is None:
                path = self._path
            self._data = self.device.file_to_dataframe(path)


class SoftwareEvent(DataStream):
    """Represents a generic Software event."""

    def __init__(self, path: Optional[str | Path] = None, **kwargs):
        super().__init__(
            path=path, **kwargs,
            data_type=DataStreamType.SOFTWARE_EVENT,
            reader=None,
            parser=None,
            )

    def _load_single_event(self, value: str) -> None:
        self._data = json.loads(value)
        return self._data

    def load_from_file(self,
                       path: Optional[str | Path] = None,
                       force_reload: bool = False) -> None:
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
        if self._data is None:
            self.load_from_file()
        df = pd.concat(
            [self._data,
            pd.json_normalize(self._data["data"]).set_index(self._data.index)],
            axis=1
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