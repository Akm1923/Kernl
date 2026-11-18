from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, Any


class GGUFError(ValueError):
    """Raised when a GGUF file is malformed or unsupported."""


@dataclass(frozen=True)
class GGUFMetadata:
    path: Path
    version: int
    tensor_count: int
    metadata_count: int
    architecture: str | None
    parameter_count: int | None
    context_length: int | None
    embedding_length: int | None
    block_count: int | None
    quantization_version: int | None
    file_size_bytes: int
    metadata: dict[str, Any]


class GGUFReader:
    MAGIC = b"GGUF"
    MAX_METADATA_ENTRIES = 100_000
    MAX_STRING_BYTES = 16 * 1024 * 1024

    def read(self, path: Path) -> GGUFMetadata:
        if not path.is_file():
            raise GGUFError(f"GGUF file not found: {path}")
        try:
            with path.open("rb") as handle:
                magic = self._read_exact(handle, 4)
                if magic != self.MAGIC:
                    raise GGUFError("file does not start with the GGUF magic")
                version = self._u32(handle)
                tensor_count = self._u64(handle)
                metadata_count = self._u64(handle)
                if version not in {1, 2, 3}:
                    raise GGUFError(f"unsupported GGUF version: {version}")
                if metadata_count > self.MAX_METADATA_ENTRIES:
                    raise GGUFError("metadata entry count is unreasonably large")
                metadata = {self._string(handle): self._value(handle) for _ in range(metadata_count)}
        except (OSError, EOFError, struct.error) as exc:
            raise GGUFError(f"could not read GGUF metadata: {exc}") from exc

        architecture = self._as_string(metadata.get("general.architecture"))
        prefix = f"{architecture}." if architecture else ""
        context_length = self._as_int(metadata.get(f"{prefix}context_length"))
        embedding_length = self._as_int(metadata.get(f"{prefix}embedding_length"))
        block_count = self._as_int(metadata.get(f"{prefix}block_count"))
        parameter_count = self._as_int(metadata.get("general.parameter_count"))
        quantization_version = self._as_int(metadata.get("general.quantization_version"))
        return GGUFMetadata(
            path=path,
            version=version,
            tensor_count=tensor_count,
            metadata_count=metadata_count,
            architecture=architecture,
            parameter_count=parameter_count,
            context_length=context_length,
            embedding_length=embedding_length,
            block_count=block_count,
            quantization_version=quantization_version,
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
        )

    def _value(self, handle: BinaryIO) -> Any:
        value_type = self._u32(handle)
        if value_type == 0:
            return self._u8(handle) != 0
        if value_type == 1:
            return self._u8(handle)
        if value_type == 2:
            return self._u16(handle)
        if value_type == 3:
            return self._u32(handle)
        if value_type == 4:
            return self._u64(handle)
        if value_type == 5:
            return self._i8(handle)
        if value_type == 6:
            return self._i16(handle)
        if value_type == 7:
            return self._i32(handle)
        if value_type == 8:
            return self._i64(handle)
        if value_type == 9:
            return self._f32(handle)
        if value_type == 10:
            return self._f64(handle)
        if value_type == 11:
            return self._string(handle)
        if value_type == 12:
            length = self._u64(handle)
            if length > self.MAX_METADATA_ENTRIES:
                raise GGUFError("array length is unreasonably large")
            element_type = self._u32(handle)
            return [self._value_with_type(handle, element_type) for _ in range(length)]
        raise GGUFError(f"unsupported GGUF metadata value type: {value_type}")

    def _value_with_type(self, handle: BinaryIO, value_type: int) -> Any:
        # Array entries carry their type once, followed by raw values.
        if value_type == 11:
            return self._string(handle)
        if value_type == 12:
            raise GGUFError("nested metadata arrays are not supported")
        return self._value_payload(handle, value_type)

    def _value_payload(self, handle: BinaryIO, value_type: int) -> Any:
        sizes = {0: ("u8", 1), 1: ("u8", 1), 2: ("u16", 2), 3: ("u32", 4), 4: ("u64", 8), 5: ("i8", 1), 6: ("i16", 2), 7: ("i32", 4), 8: ("i64", 8), 9: ("f32", 4), 10: ("f64", 8)}
        if value_type not in sizes:
            raise GGUFError(f"unsupported array value type: {value_type}")
        return getattr(self, "_" + sizes[value_type][0])(handle)

    def _string(self, handle: BinaryIO) -> str:
        length = self._u64(handle)
        if length > self.MAX_STRING_BYTES:
            raise GGUFError("GGUF string is unreasonably large")
        return self._read_exact(handle, length).decode("utf-8", errors="replace")

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @staticmethod
    def _as_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _read_exact(handle: BinaryIO, size: int) -> bytes:
        data = handle.read(size)
        if len(data) != size:
            raise EOFError("unexpected end of file")
        return data

    def _u8(self, h): return struct.unpack("<B", self._read_exact(h, 1))[0]
    def _u16(self, h): return struct.unpack("<H", self._read_exact(h, 2))[0]
    def _u32(self, h): return struct.unpack("<I", self._read_exact(h, 4))[0]
    def _u64(self, h): return struct.unpack("<Q", self._read_exact(h, 8))[0]
    def _i8(self, h): return struct.unpack("<b", self._read_exact(h, 1))[0]
    def _i16(self, h): return struct.unpack("<h", self._read_exact(h, 2))[0]
    def _i32(self, h): return struct.unpack("<i", self._read_exact(h, 4))[0]
    def _i64(self, h): return struct.unpack("<q", self._read_exact(h, 8))[0]
    def _f32(self, h): return struct.unpack("<f", self._read_exact(h, 4))[0]
    def _f64(self, h): return struct.unpack("<d", self._read_exact(h, 8))[0]
