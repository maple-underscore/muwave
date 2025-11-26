#!/bin/bash
# muwave quickstart script
# This script sets up everything you need to get started with muwave.

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      muwave Quickstart                          ║"
echo "║     Sound-based communication protocol for AI agents           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check for Python 3.9+
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
        MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
            echo "✓ Python $PYTHON_VERSION found"
            return 0
        else
            echo "✗ Python 3.9+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        echo "✗ Python 3 not found"
        return 1
    fi
}

# Check for pip
check_pip() {
    if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        echo "✓ pip found"
        return 0
    else
        echo "✗ pip not found"
        return 1
    fi
}

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Checking prerequisites..."
echo ""

check_python || { echo "Please install Python 3.9 or higher"; exit 1; }
check_pip || { echo "Please install pip"; exit 1; }

echo ""
echo "Installing muwave..."
echo ""

# Install muwave in editable mode
cd "$SCRIPT_DIR"
pip3 install -e . --quiet

echo "✓ muwave installed successfully"
echo ""

# Initialize configuration if it doesn't exist
if [ ! -f "$SCRIPT_DIR/config.yaml" ]; then
    echo "Initializing configuration..."
    muwave init
    echo "✓ Configuration created at config.yaml"
else
    echo "✓ Configuration already exists at config.yaml"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now use muwave with the following commands:"
echo ""
echo "  Interactive mode:"
echo "    muwave run --name \"Alice\""
echo ""
echo "  Send a message:"
echo "    muwave send \"Hello, World!\""
echo ""
echo "  Generate a sound wave from text and save to WAV:"
echo "    muwave generate \"Hello, World!\" -o hello.wav"
echo ""
echo "  Listen for messages:"
echo "    muwave listen --timeout 30"
echo ""
echo "  Send to AI (requires Ollama):"
echo "    muwave ai \"What is the capital of France?\" --transmit"
echo ""
echo "  Start web interface:"
echo "    muwave web --port 5000"
echo ""
echo "  List available audio devices:"
echo "    muwave devices"
echo ""
echo "For more information, see the README.md file."
echo ""
