"""Clear data-layer exceptions."""


class DataError(Exception):
    """Base class for benchmark data failures."""


class BenchmarkNotFoundError(DataError):
    """Raised when a benchmark identifier is not registered."""


class DataFileNotFoundError(DataError):
    """Raised when a required public data file is missing."""


class MissingColumnError(DataError):
    """Raised when a required column is absent."""


class ChannelRoleError(DataError):
    """Raised when channel roles overlap or disagree with the data."""


class DataAlignmentError(DataError):
    """Raised when row alignment, time, or split separation is invalid."""


class DataFormatError(DataError):
    """Raised when CSV, JSON, or JSONL content cannot be decoded."""


class ScalingError(DataError):
    """Raised for invalid or leaking scaling operations."""

