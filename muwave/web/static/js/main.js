/**
 * muwave Web Interface JavaScript
 * Real-time audio visualization and communication
 */

// Global socket connection
let socket = null;

// Canvas contexts
let waveformCtx = null;
let spectrogramCtx = null;

// Animation frame IDs
let waveformAnimationId = null;
let spectrogramAnimationId = null;

// Data buffers
let waveformData = [];
let spectrogramData = [];

// Colors
const COLORS = {
    waveform: '#00BCD4',
    waveformBg: 'rgba(0, 0, 0, 0.3)',
    spectrogramLow: '#1a1a2e',
    spectrogramHigh: '#00BCD4',
    grid: 'rgba(255, 255, 255, 0.1)'
};

/**
 * Initialize Socket.IO connection
 */
function initSocket() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to muwave server');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });
    
    socket.on('stats_update', function(data) {
        updateStats(data);
    });
    
    socket.on('waveform_update', function(data) {
        if (data.waveform) {
            waveformData = data.waveform;
        }
    });
    
    socket.on('spectrogram_update', function(data) {
        if (data.spectrogram) {
            spectrogramData = data.spectrogram;
        }
    });
    
    socket.on('parties_update', function(data) {
        updateParties(data);
    });
    
    socket.on('message_received', function(data) {
        showMessageNotification(data);
    });
    
    socket.on('transmission_progress', function(data) {
        console.log('Transmission progress:', data);
        // Find the last sent message and update its style during transmission
        const messages = document.querySelectorAll('.message-sent');
        const lastMessage = messages[messages.length - 1];
        if (lastMessage) {
            if (data.status === 'sending') {
                lastMessage.style.borderLeftColor = '#00ff00';
                lastMessage.style.backgroundColor = 'rgba(0, 255, 0, 0.05)';
                lastMessage.style.transition = 'all 0.3s ease';
            } else if (data.status === 'sent') {
                lastMessage.style.borderLeftColor = '#00aa00';
                lastMessage.style.backgroundColor = '';
            } else if (data.status === 'error') {
                lastMessage.style.borderLeftColor = '#ff0000';
                lastMessage.style.backgroundColor = 'rgba(255, 0, 0, 0.05)';
            }
        }
    });
}

/**
 * Update system statistics display
 */
function updateStats(stats) {
    // CPU
    const cpuBar = document.getElementById('cpu-bar');
    const cpuValue = document.getElementById('cpu-value');
    if (cpuBar && cpuValue) {
        cpuBar.style.width = stats.cpu_percent + '%';
        cpuValue.textContent = stats.cpu_percent.toFixed(1) + '%';
    }
    
    // RAM
    const ramBar = document.getElementById('ram-bar');
    const ramValue = document.getElementById('ram-value');
    if (ramBar && ramValue) {
        ramBar.style.width = stats.ram_percent + '%';
        ramValue.textContent = stats.ram_percent.toFixed(1) + '% (' + 
                              stats.ram_used_gb.toFixed(1) + '/' + 
                              stats.ram_total_gb.toFixed(1) + ' GB)';
    }
    
    // GPU
    const gpuBar = document.getElementById('gpu-bar');
    const gpuValue = document.getElementById('gpu-value');
    if (gpuBar && gpuValue) {
        if (stats.gpu_percent > 0 || stats.gpu_memory_total_gb > 0) {
            gpuBar.style.width = stats.gpu_percent + '%';
            gpuValue.textContent = stats.gpu_percent.toFixed(1) + '% (' + 
                                  stats.gpu_memory_used_gb.toFixed(1) + '/' + 
                                  stats.gpu_memory_total_gb.toFixed(1) + ' GB)';
        } else {
            gpuBar.style.width = '0%';
            gpuValue.textContent = 'N/A';
        }
    }
}

/**
 * Initialize waveform visualization
 */
function initWaveform(canvasId, color = null) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    waveformCtx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    waveformCtx.scale(2, 2);
    
    const waveformColor = color || COLORS.waveform;
    
    function drawWaveform() {
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        const centerY = height / 2;
        
        // Clear canvas
        waveformCtx.fillStyle = COLORS.waveformBg;
        waveformCtx.fillRect(0, 0, width, height);
        
        // Draw grid
        waveformCtx.strokeStyle = COLORS.grid;
        waveformCtx.lineWidth = 1;
        waveformCtx.beginPath();
        waveformCtx.moveTo(0, centerY);
        waveformCtx.lineTo(width, centerY);
        waveformCtx.stroke();
        
        // Draw waveform
        if (waveformData.length > 0) {
            waveformCtx.strokeStyle = waveformColor;
            waveformCtx.lineWidth = 2;
            waveformCtx.beginPath();
            
            const step = width / waveformData.length;
            
            for (let i = 0; i < waveformData.length; i++) {
                const x = i * step;
                const y = centerY + (waveformData[i] * centerY * 0.8);
                
                if (i === 0) {
                    waveformCtx.moveTo(x, y);
                } else {
                    waveformCtx.lineTo(x, y);
                }
            }
            
            waveformCtx.stroke();
            
            // Add glow effect
            waveformCtx.shadowColor = waveformColor;
            waveformCtx.shadowBlur = 10;
            waveformCtx.stroke();
            waveformCtx.shadowBlur = 0;
        }
        
        waveformAnimationId = requestAnimationFrame(drawWaveform);
    }
    
    drawWaveform();
}

/**
 * Initialize spectrogram visualization
 */
function initSpectrogram(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    spectrogramCtx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    spectrogramCtx.scale(2, 2);
    
    function drawSpectrogram() {
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        
        // Clear canvas
        spectrogramCtx.fillStyle = COLORS.spectrogramLow;
        spectrogramCtx.fillRect(0, 0, width, height);
        
        if (spectrogramData.length > 0) {
            const columnWidth = width / spectrogramData.length;
            
            for (let i = 0; i < spectrogramData.length; i++) {
                const column = spectrogramData[i];
                if (!column) continue;
                
                const binHeight = height / column.length;
                
                for (let j = 0; j < column.length; j++) {
                    const value = column[j] || 0;
                    const color = getSpectrogramColor(value);
                    
                    spectrogramCtx.fillStyle = color;
                    spectrogramCtx.fillRect(
                        i * columnWidth,
                        height - (j + 1) * binHeight,
                        columnWidth + 1,
                        binHeight + 1
                    );
                }
            }
        }
        
        // Draw frequency labels
        spectrogramCtx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        spectrogramCtx.font = '10px sans-serif';
        spectrogramCtx.fillText('22kHz', 5, 15);
        spectrogramCtx.fillText('0Hz', 5, height - 5);
        
        spectrogramAnimationId = requestAnimationFrame(drawSpectrogram);
    }
    
    drawSpectrogram();
}

/**
 * Get color for spectrogram value (0-1)
 */
function getSpectrogramColor(value) {
    // Gradient from dark to cyan
    const r = Math.floor(26 + (0 - 26) * value);
    const g = Math.floor(26 + (188 - 26) * value);
    const b = Math.floor(46 + (212 - 46) * value);
    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Update parties list
 */
function updateParties(parties) {
    const list = document.getElementById('parties-list');
    if (!list) return;
    
    list.innerHTML = '';
    
    if (parties.length === 0) {
        list.innerHTML = '<p>No active parties</p>';
        return;
    }
    
    parties.forEach(party => {
        const div = document.createElement('div');
        div.className = 'party-item';
        div.style.borderLeft = `4px solid ${party.color}`;
        
        let status = '';
        if (party.is_speaking) status += '🔊 ';
        if (party.is_listening) status += '👂';
        
        div.innerHTML = `
            <span class="party-name">${escapeHtml(party.name)}</span>
            <span class="party-status">${status}</span>
        `;
        
        list.appendChild(div);
    });
}

/**
 * Show message notification
 */
function showMessageNotification(data) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.innerHTML = `
        <strong>Message from ${data.sender || 'Unknown'}</strong>
        <p>${escapeHtml(data.content.substring(0, 100))}${data.content.length > 100 ? '...' : ''}</p>
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Refresh stats manually
 */
function refreshStats() {
    if (socket) {
        socket.emit('request_stats');
    }
}

/**
 * Update queue status display
 */
function updateQueueStatus(data) {
    const queueList = document.getElementById('queue-list');
    if (!queueList) return;
    
    // Remove "empty" message if present
    const empty = queueList.querySelector('.queue-empty');
    if (empty) empty.remove();
    
    // Add new queue item
    const item = document.createElement('div');
    item.className = 'queue-item';
    item.innerHTML = `
        <span class="queue-text">${escapeHtml(data.text.substring(0, 30))}...</span>
        <span class="queue-status pending">Pending</span>
    `;
    queueList.appendChild(item);
}

/**
 * Cleanup on page unload
 */
window.addEventListener('beforeunload', function() {
    if (waveformAnimationId) {
        cancelAnimationFrame(waveformAnimationId);
    }
    if (spectrogramAnimationId) {
        cancelAnimationFrame(spectrogramAnimationId);
    }
    if (socket) {
        socket.disconnect();
    }
});

// Handle window resize
window.addEventListener('resize', function() {
    // Reinitialize canvases on resize
    const waveformCanvas = document.getElementById('waveform-canvas');
    const spectrogramCanvas = document.getElementById('spectrogram-canvas');
    
    if (waveformCanvas && waveformCtx) {
        waveformCanvas.width = waveformCanvas.offsetWidth * 2;
        waveformCanvas.height = waveformCanvas.offsetHeight * 2;
        waveformCtx.scale(2, 2);
    }
    
    if (spectrogramCanvas && spectrogramCtx) {
        spectrogramCanvas.width = spectrogramCanvas.offsetWidth * 2;
        spectrogramCanvas.height = spectrogramCanvas.offsetHeight * 2;
        spectrogramCtx.scale(2, 2);
    }
});
