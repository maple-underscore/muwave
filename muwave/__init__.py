"""
muwave - Sound-based communication protocol for AI agents
Compatible with Linux and macOS
"""

__version__ = "0.1.5"
__author__ = "muwave contributors"

from muwave.core.config import Config
from muwave.core.party import Party
from muwave.protocol.transmitter import Transmitter
from muwave.protocol.receiver import Receiver

__all__ = [
    "Config",
    "Party",
    "Transmitter",
    "Receiver",
    "__version__",
]
