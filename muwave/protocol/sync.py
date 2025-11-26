"""
Process synchronization for muwave.
Handles coordination between multiple processes on the same machine.
"""

import os
import time
import fcntl
import threading
import socket
from pathlib import Path
from typing import Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


class SyncMethod(Enum):
    """Synchronization method."""
    FILE = "file"
    SOCKET = "socket"


@dataclass
class SyncState:
    """Current synchronization state."""
    audio_active: bool = False
    active_party_id: Optional[str] = None
    timestamp: float = 0.0
    text_pending: bool = False


class ProcessSync:
    """
    Process synchronization manager.
    
    Coordinates between multiple muwave processes on the same machine
    to prevent audio collisions and manage text input timing.
    """
    
    def __init__(
        self,
        party_id: str,
        method: SyncMethod = SyncMethod.FILE,
        lock_file: str = "/tmp/muwave_sync.lock",
        socket_port: int = 19876,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize process sync.
        
        Args:
            party_id: Unique identifier for this party
            method: Synchronization method (file or socket)
            lock_file: Path to lock file (for file method)
            socket_port: Port for socket-based sync
            timeout_seconds: Maximum wait time for sync
        """
        self.party_id = party_id
        self.method = method
        self.lock_file = Path(lock_file)
        self.socket_port = socket_port
        self.timeout_seconds = timeout_seconds
        
        self._state = SyncState()
        self._lock = threading.Lock()
        self._file_lock: Optional[int] = None
        self._socket: Optional[socket.socket] = None
        
        self._ensure_lock_file()
    
    def _ensure_lock_file(self) -> None:
        """Ensure lock file exists."""
        if self.method == SyncMethod.FILE:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.lock_file.exists():
                self.lock_file.touch()
    
    @contextmanager
    def audio_lock(self):
        """
        Context manager for exclusive audio access.
        
        Usage:
            with sync.audio_lock():
                # Transmit or receive audio
                pass
        """
        acquired = self.acquire_audio_lock()
        try:
            yield acquired
        finally:
            if acquired:
                self.release_audio_lock()
    
    def acquire_audio_lock(self) -> bool:
        """
        Acquire exclusive audio lock.
        
        Returns:
            True if lock acquired successfully
        """
        if self.method == SyncMethod.FILE:
            return self._acquire_file_lock()
        else:
            return self._acquire_socket_lock()
    
    def release_audio_lock(self) -> None:
        """Release audio lock."""
        if self.method == SyncMethod.FILE:
            self._release_file_lock()
        else:
            self._release_socket_lock()
    
    def _acquire_file_lock(self) -> bool:
        """Acquire file-based lock."""
        try:
            fd = os.open(str(self.lock_file), os.O_RDWR | os.O_CREAT)
            start_time = time.time()
            
            while time.time() - start_time < self.timeout_seconds:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._file_lock = fd
                    
                    # Write our party ID to the file
                    os.ftruncate(fd, 0)
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.write(fd, f"{self.party_id}:{time.time()}".encode())
                    
                    with self._lock:
                        self._state.audio_active = True
                        self._state.active_party_id = self.party_id
                        self._state.timestamp = time.time()
                    
                    return True
                except OSError:
                    time.sleep(0.1)
            
            os.close(fd)
            return False
            
        except Exception:
            return False
    
    def _release_file_lock(self) -> None:
        """Release file-based lock."""
        if self._file_lock is not None:
            try:
                fcntl.flock(self._file_lock, fcntl.LOCK_UN)
                os.close(self._file_lock)
            except Exception:
                pass
            finally:
                self._file_lock = None
                with self._lock:
                    self._state.audio_active = False
                    self._state.active_party_id = None
    
    def _acquire_socket_lock(self) -> bool:
        """Acquire socket-based lock."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            start_time = time.time()
            while time.time() - start_time < self.timeout_seconds:
                try:
                    self._socket.bind(('127.0.0.1', self.socket_port))
                    with self._lock:
                        self._state.audio_active = True
                        self._state.active_party_id = self.party_id
                        self._state.timestamp = time.time()
                    return True
                except OSError:
                    time.sleep(0.1)
            
            return False
            
        except Exception:
            return False
    
    def _release_socket_lock(self) -> None:
        """Release socket-based lock."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            finally:
                self._socket = None
                with self._lock:
                    self._state.audio_active = False
                    self._state.active_party_id = None
    
    def is_audio_active(self) -> bool:
        """Check if any process is currently using audio."""
        if self.method == SyncMethod.FILE:
            return self._check_file_lock()
        else:
            return self._check_socket_lock()
    
    def _check_file_lock(self) -> bool:
        """Check if file lock is held."""
        try:
            fd = os.open(str(self.lock_file), os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False  # Lock was available
            except OSError:
                return True  # Lock is held
            finally:
                os.close(fd)
        except Exception:
            return False
    
    def _check_socket_lock(self) -> bool:
        """Check if socket lock is held."""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                test_socket.bind(('127.0.0.1', self.socket_port))
                return False  # Port was available
            except OSError:
                return True  # Port is in use
            finally:
                test_socket.close()
        except Exception:
            return False
    
    def wait_for_audio_complete(
        self,
        callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Wait for any active audio process to complete.
        
        Args:
            callback: Optional callback when audio is complete
            
        Returns:
            True if completed within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < self.timeout_seconds:
            if not self.is_audio_active():
                if callback:
                    callback()
                return True
            time.sleep(0.1)
        
        return False
    
    def set_text_pending(self, pending: bool = True) -> None:
        """Mark that text input is pending."""
        with self._lock:
            self._state.text_pending = pending
    
    def is_text_pending(self) -> bool:
        """Check if text input is pending."""
        with self._lock:
            return self._state.text_pending
    
    def get_state(self) -> SyncState:
        """Get current sync state."""
        with self._lock:
            return SyncState(
                audio_active=self._state.audio_active,
                active_party_id=self._state.active_party_id,
                timestamp=self._state.timestamp,
                text_pending=self._state.text_pending,
            )
    
    def get_active_party_id(self) -> Optional[str]:
        """Get the ID of the party currently holding the audio lock."""
        if self.method == SyncMethod.FILE and self.lock_file.exists():
            try:
                content = self.lock_file.read_text()
                if ':' in content:
                    return content.split(':')[0]
            except Exception:
                pass
        return None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.release_audio_lock()
