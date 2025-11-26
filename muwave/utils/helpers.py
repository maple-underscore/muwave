"""
Utility functions for muwave.
"""

import platform
import subprocess
from typing import Optional, Tuple


def get_platform() -> str:
    """Get the current platform (linux or darwin for macOS)."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system


def is_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5.0,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_ollama_container_running(container_name: str = "ollama") -> bool:
    """Check if the Ollama Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return container_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_ollama_container(
    container_name: str = "ollama",
    port: int = 11434,
) -> Tuple[bool, str]:
    """
    Start the Ollama Docker container.
    
    Args:
        container_name: Name for the container
        port: Port to expose
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Check if container already exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        
        if container_name in result.stdout:
            # Container exists, try to start it
            result = subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            if result.returncode == 0:
                return True, f"Started existing container '{container_name}'"
            return False, f"Failed to start container: {result.stderr}"
        
        # Create and start new container
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", f"{port}:11434",
                "-v", "ollama:/root/.ollama",
                "ollama/ollama",
            ],
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        
        if result.returncode == 0:
            return True, f"Created and started container '{container_name}'"
        return False, f"Failed to create container: {result.stderr}"
        
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out"
    except FileNotFoundError:
        return False, "Docker not found"


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"
