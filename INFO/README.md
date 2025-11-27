# muwave

Sound-based communication protocol for AI agents, compatible with Linux and macOS.

## Features

- **Multiple Transmission Modes**: Slow, medium, and fast transmission speeds with configurable symbol duration
- **Redundancy Options**: Low, medium, and high redundancy with error correction
- **Multi-Party Support**: Run multiple parties on the same machine with different audio tones
- **Ollama Integration**: Connect to Ollama via Docker, terminal (`ollama run`), or HTTP API with conversation context
- **External Party Support**: Protocol allows external parties to join the conversation
- **Text Input Synchronization**: Wait for audio processes to complete before processing text input
- **Self-Recognition**: Parties can recognize their own output, even on the same machine
- **Distinctive Audio Signals**: Rising chirp for start, falling chirp for end of transmission
- **Rich Text Interface**: Color-coded transmission status (yellow=waiting, blue=sending, green=sent)
- **Real-time Receiving Display**: See incoming messages as they're decoded
- **Web Interface**: Browser-based UI with audio waveform and spectrogram visualization
- **Conversation Logging**: Save all communications to a log file
- **System Monitoring**: CPU, RAM, and GPU usage tracking
- **Easy Configuration**: YAML-based configuration for all settings

## Installation

```bash
# Clone the repository
git clone https://github.com/maple-underscore/muwave.git
cd muwave

# Install with pip
pip install -e .

# Or install with optional dependencies for development
pip install -e ".[dev]"
```

### Dependencies

- Python 3.9+
- numpy, scipy for signal processing
- sounddevice for audio I/O
- rich for terminal interface
- Flask, flask-socketio for web interface
- psutil for system monitoring
- requests for Ollama HTTP API
- PyYAML for configuration

## Quick Start

### Initialize Configuration

```bash
muwave init
```

This creates a `config.yaml` file with default settings.

### Run Interactive Mode

```bash
muwave run --name "Alice"
```

### Send a Single Message

```bash
muwave send "Hello, World!"
```

### Generate Sound Wave and Save to File

```bash
muwave generate "Hello, World!" -o hello.wav
```

This encodes the text into an FSK-modulated audio signal and saves it as a WAV file. The file can be downloaded and played on any device.

### Listen for Messages

```bash
muwave listen --timeout 30
```

### Send to AI

```bash
muwave ai "What is the capital of France?" --transmit
```

### Start Web Interface

```bash
muwave web --port 5000
```

Then open http://localhost:5000 in your browser.

### Basic Usage

Generate and decode messages:
- Check prerequisites (Python 3.9+, pip)
- Install muwave and its dependencies
- Initialize a configuration file
- Display available commands

## Configuration

Edit `config.yaml` to customize muwave:

```yaml
# Transmission speed
speed:
  mode: medium  # slow, medium, fast
  modes:
    slow:
      symbol_duration_ms: 100
    medium:
      symbol_duration_ms: 50
    fast:
      symbol_duration_ms: 25

# Redundancy settings
redundancy:
  mode: medium  # low, medium, high
  modes:
    low:
      repetitions: 1
      error_correction: false
    medium:
      repetitions: 2
      error_correction: true
    high:
      repetitions: 3
      error_correction: true

# Ollama integration
ollama:
  mode: docker  # docker, terminal, http
  docker:
    container_name: "ollama"
    host: "localhost"
    port: 11434
  model:
    name: "llama3.2"
    keep_context: true
```

## Ollama Integration

### Docker Mode

Start Ollama in Docker:

```bash
muwave docker-start
```

This creates and starts an Ollama container.

### Terminal Mode

Use the `ollama run` command directly:

```yaml
ollama:
  mode: terminal
  model:
    name: "llama3.2"
```

### HTTP API Mode

Connect to a running Ollama instance:

```yaml
ollama:
  mode: http
  http:
    base_url: "http://localhost:11434"
```

### Conversation Context

muwave maintains conversation context across messages:

```python
from muwave.ollama.client import OllamaClient, OllamaConfig

client = OllamaClient(OllamaConfig(
    model="llama3.2",
    keep_context=True,
))

# First message
response1 = client.chat("What is the capital of France?")
# Response: "The capital of France is Paris."

# Second message (with context)
response2 = client.chat("What is its population?")
# Response: "Paris has a population of approximately 2.1 million..."
```

## Web Interface

The web interface provides:

- **Dashboard**: System stats, party list, audio visualization
- **User Input**: Compose and send messages with AI support
- **Party Pages**: Individual views for each party with conversation log

### Starting the Web Server

```bash
muwave web --port 5000 --party "Alice" --party "Bob"
```

### Endpoints

- `/` - Dashboard with system overview
- `/user` - User input interface
- `/party/<party_id>` - Party-specific interface

### Features

- Real-time audio waveform visualization
- Spectrogram history display
- Message queue management
- CPU/RAM/GPU monitoring
- Different audio tones for different parties

## Multi-Party Communication

Run multiple parties on the same machine:

```bash
# Terminal 1
muwave run --name "Alice"

# Terminal 2
muwave run --name "Bob"
```

Each party has a unique signature and can recognize its own transmissions to avoid feedback loops.

## Protocol Details

### FSK Modulation

muwave uses Frequency-Shift Keying (FSK) to encode data:

- 16 frequency bins for encoding (4 bits per symbol)
- Each byte encoded as two symbols (nibbles)
- Configurable symbol duration for speed/reliability tradeoff

### Transmission Format

1. **Start Signal**: Rising chirp (500 Hz → 800 Hz)
2. **Party Signature**: 8 bytes identifying the sender
3. **Data Length**: 2 bytes (max 65535 bytes)
4. **Data**: Encoded message (with optional repetitions)
5. **End Signal**: Falling chirp (900 Hz → 600 Hz)

### Self-Recognition

Each party generates a unique 8-byte signature based on:
- UUID
- Process ID
- Timestamp

This allows parties to filter out their own transmissions when listening.

## API Reference

### Party

```python
from muwave.core.party import Party

party = Party(name="Alice", is_ai=False)
party.set_system_prompt("You are a helpful assistant.")

message = party.create_message("Hello!")
party.mark_transmitted(message)
```

### Transmitter

```python
from muwave.protocol.transmitter import Transmitter

tx = Transmitter(party, config={
    "speed_mode": "medium",
    "repetitions": 2,
})

tx.transmit_text("Hello, World!")
```

### Receiver

```python
from muwave.protocol.receiver import Receiver

rx = Receiver(party)
rx.set_message_callback(lambda msg, is_own: print(f"Received: {msg.content}"))
rx.start_listening()
```

### Ollama Client

```python
from muwave.ollama.client import OllamaClient, OllamaConfig, OllamaMode

client = OllamaClient(OllamaConfig(
    mode=OllamaMode.DOCKER,
    model="llama3.2",
    keep_context=True,
))

response = client.chat("Hello!")
print(response)
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black muwave/
isort muwave/
```

### Type Checking

```bash
mypy muwave/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Credits

Inspired by [ggwave](https://github.com/ggerganov/ggwave) and [gibberlink](https://github.com/PennyroyalTea/gibberlink).
