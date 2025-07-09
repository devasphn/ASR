from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import numpy as np
import torch
import whisperx
import gc
import logging
from typing import Dict, List
import base64
import io
from pysilero_vad import SileroVoiceActivityDetector
import queue
import threading
import time
from datetime import datetime
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
batch_size = 16
model = None
align_model = None
metadata = None
vad_detector = None

# Initialize models
def initialize_models():
    global model, align_model, metadata, vad_detector
    
    try:
        # Initialize WhisperX model with language specification
        logger.info("Loading WhisperX model...")
        model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="en")
        
        # Initialize alignment model
        logger.info("Loading alignment model...")
        align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
        
        # Initialize Silero VAD with lower threshold
        logger.info("Loading Silero VAD...")
        vad_detector = SileroVoiceActivityDetector()
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

# Modern lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_models()
    yield
    # Shutdown (cleanup if needed)
    if device == "cuda":
        torch.cuda.empty_cache()

# Initialize FastAPI app with lifespan
app = FastAPI(title="Real-Time STT API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, List[np.ndarray]] = {}
        self.transcription_tasks: Dict[str, asyncio.Task] = {}
        self.last_transcription_time: Dict[str, float] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.last_transcription_time[client_id] = time.time()
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.transcription_tasks:
            self.transcription_tasks[client_id].cancel()
            del self.transcription_tasks[client_id]
        if client_id in self.last_transcription_time:
            del self.last_transcription_time[client_id]
        logger.info(f"Client {client_id} disconnected")
        
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                
    def add_audio_chunk(self, client_id: str, audio_array: np.ndarray):
        if client_id in self.audio_buffers:
            self.audio_buffers[client_id].append(audio_array)

manager = ConnectionManager()

# Enhanced audio processing functions
def process_audio_chunk(audio_data: bytes) -> np.ndarray:
    """Process raw PCM audio data with proper normalization"""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Convert raw PCM bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 and normalize properly
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Debug: Calculate RMS to verify audio signal
        rms = np.sqrt(np.mean(audio_array**2))
        if rms > 0.001:  # Only log when there's actual audio
            logger.debug(f"Audio chunk processed: {len(audio_array)} samples, RMS: {rms:.4f}")
        
        return audio_array
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        return np.array([])

def detect_voice_activity(audio_array: np.ndarray) -> bool:
    """Detect voice activity using Silero VAD with enhanced sensitivity"""
    try:
        if len(audio_array) == 0:
            return False
        
        # Calculate RMS first for quick energy check
        rms = np.sqrt(np.mean(audio_array**2))
        if rms < 0.001:  # Very low energy threshold
            return False
            
        # Ensure we have enough samples for VAD
        min_samples = 512
        if len(audio_array) < min_samples:
            audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))
        
        # Process in chunks for better VAD performance
        chunk_size = 512
        vad_scores = []
        
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i:i + chunk_size]
            
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Convert to int16 for VAD with proper scaling
            audio_int16 = np.clip(chunk * 32767, -32767, 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Get VAD score
            vad_score = vad_detector(audio_bytes)
            vad_scores.append(vad_score)
        
        # Use lower threshold for better sensitivity
        avg_vad_score = np.mean(vad_scores) if vad_scores else 0.0
        has_voice = avg_vad_score >= 0.2  # Lowered from 0.3 to 0.2
        
        if has_voice:
            logger.info(f"Voice detected! VAD score: {avg_vad_score:.3f}, RMS: {rms:.4f}")
        
        return has_voice
        
    except Exception as e:
        logger.error(f"Error in VAD detection: {e}")
        return False

async def transcribe_audio(audio_array: np.ndarray) -> str:
    """Transcribe audio using WhisperX with improved preprocessing"""
    try:
        if len(audio_array) == 0:
            return ""
            
        # Ensure minimum audio length (1.5 seconds for better accuracy)
        min_length = 24000  # 1.5 seconds at 16kHz
        if len(audio_array) < min_length:
            audio_array = np.pad(audio_array, (0, min_length - len(audio_array)))
            
        # Ensure audio is float32 and properly normalized
        audio_array = audio_array.astype(np.float32)
        
        # Apply gentle normalization if needed
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val * 0.95  # Normalize to 95% to avoid clipping
        
        logger.info(f"Starting transcription of {len(audio_array)} samples, max_val: {max_val:.4f}")
        
        # Transcribe with WhisperX
        result = model.transcribe(audio_array, batch_size=batch_size)
        
        # Align transcript if alignment model is available
        if align_model is not None and result.get("segments"):
            try:
                result = whisperx.align(result["segments"], align_model, metadata, 
                                     audio_array, device, return_char_alignments=False)
            except Exception as e:
                logger.warning(f"Alignment failed: {e}, using original result")
        
        # Extract text from segments
        text = ""
        if "segments" in result:
            for segment in result["segments"]:
                if "text" in segment:
                    text += segment["text"] + " "
                    
        final_text = text.strip()
        logger.info(f"Transcription completed: '{final_text}'")
        return final_text
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return ""

# Background transcription task
async def continuous_transcription(client_id: str):
    """Continuously process audio chunks for transcription"""
    
    while client_id in manager.active_connections:
        try:
            # Get current audio buffer
            current_buffer = manager.audio_buffers.get(client_id, [])
            
            if len(current_buffer) == 0:
                await asyncio.sleep(0.2)  # Check every 200ms
                continue
                
            # Process audio chunks with voice activity detection
            voice_chunks = []
            processed_chunks = 0
            
            for audio_chunk in current_buffer:
                if detect_voice_activity(audio_chunk):
                    voice_chunks.append(audio_chunk)
                processed_chunks += 1
                
            # Clear processed chunks
            if processed_chunks > 0:
                manager.audio_buffers[client_id] = current_buffer[processed_chunks:]
                
            # Check if we have enough voice data for transcription
            if len(voice_chunks) >= 3:  # At least 3 chunks with voice
                # Concatenate voice chunks
                combined_audio = np.concatenate(voice_chunks)
                
                # Check timing constraint
                current_time = time.time()
                last_time = manager.last_transcription_time.get(client_id, 0)
                
                # Transcribe if we have enough audio (at least 1.5 seconds) and enough time has passed
                if len(combined_audio) >= 24000 and (current_time - last_time) >= 1.5:
                    
                    logger.info(f"Starting transcription for client {client_id} with {len(combined_audio)} samples")
                    
                    # Transcribe
                    text = await transcribe_audio(combined_audio)
                    
                    if text and text.strip():
                        logger.info(f"Sending transcription to client {client_id}: '{text}'")
                        
                        # Send transcription result
                        await manager.send_personal_message({
                            "type": "transcription",
                            "text": text,
                            "timestamp": datetime.now().isoformat()
                        }, client_id)
                        
                        # Update last transcription time
                        manager.last_transcription_time[client_id] = current_time
                    else:
                        logger.warning(f"Empty transcription result for client {client_id}")
                        
                    # Clean up GPU memory
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                        
        except Exception as e:
            logger.error(f"Error in continuous transcription for {client_id}: {e}")
            await asyncio.sleep(1)

# API Routes
@app.get("/")
async def get_index():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time STT</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            #transcription {
                border: 1px solid #ddd;
                padding: 20px;
                min-height: 200px;
                margin: 20px 0;
                background-color: #f9f9f9;
                border-radius: 4px;
                max-height: 400px;
                overflow-y: auto;
            }
            button {
                padding: 12px 24px;
                margin: 10px;
                font-size: 16px;
                cursor: pointer;
                border: none;
                border-radius: 4px;
                transition: background-color 0.3s;
            }
            #startBtn {
                background-color: #4CAF50;
                color: white;
            }
            #startBtn:hover {
                background-color: #45a049;
            }
            #stopBtn {
                background-color: #f44336;
                color: white;
            }
            #stopBtn:hover {
                background-color: #da190b;
            }
            #status {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            .recording {
                background-color: #ffeb3b;
                color: #333;
                border-left: 4px solid #ff9800;
            }
            .stopped {
                background-color: #e0e0e0;
                color: #666;
                border-left: 4px solid #9e9e9e;
            }
            .connected {
                background-color: #c8e6c9;
                color: #2e7d32;
                border-left: 4px solid #4caf50;
            }
            .error {
                background-color: #ffcdd2;
                color: #c62828;
                border-left: 4px solid #f44336;
            }
            .audio-level {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Real-Time Speech-to-Text</h1>
            <div id="status" class="stopped">Status: Stopped</div>
            <div id="audioLevel" class="audio-level">Audio Level: 0.00</div>
            <button id="startBtn" onclick="startRecording()">Start Recording</button>
            <button id="stopBtn" onclick="stopRecording()" disabled>Stop Recording</button>
            <div id="transcription">
                <p><em>Transcribed text will appear here...</em></p>
            </div>
        </div>
        
        <script>
            let audioContext;
            let audioWorkletNode;
            let websocket;
            let clientId = 'client_' + Date.now();
            let isRecording = false;
            let audioChunkCounter = 0;
            
            // AudioWorklet processor code with enhanced audio processing
            const audioWorkletCode = `
                class AudioProcessor extends AudioWorkletProcessor {
                    constructor() {
                        super();
                        this.chunkCounter = 0;
                        console.log('AudioWorklet processor initialized');
                    }
                    
                    process(inputs, outputs, parameters) {
                        const input = inputs[0];
                        if (input.length > 0) {
                            const inputChannel = input[0];
                            
                            // Apply gain to boost signal
                            const gainFactor = 2.0; // Increase gain
                            const processedChannel = new Float32Array(inputChannel.length);
                            
                            for (let i = 0; i < inputChannel.length; i++) {
                                processedChannel[i] = Math.max(-1, Math.min(1, inputChannel[i] * gainFactor));
                            }
                            
                            // Convert float32 to int16
                            const int16Array = new Int16Array(processedChannel.length);
                            for (let i = 0; i < processedChannel.length; i++) {
                                int16Array[i] = Math.max(-32768, Math.min(32767, processedChannel[i] * 32767));
                            }
                            
                            // Calculate RMS for debugging
                            let rms = 0;
                            for (let i = 0; i < processedChannel.length; i++) {
                                rms += processedChannel[i] * processedChannel[i];
                            }
                            rms = Math.sqrt(rms / processedChannel.length);
                            
                            // Send to main thread with debug info
                            this.port.postMessage({
                                type: 'audio',
                                data: int16Array,
                                rms: rms,
                                counter: this.chunkCounter++
                            });
                        }
                        return true;
                    }
                }
                
                registerProcessor('audio-processor', AudioProcessor);
            `;
            
            // Auto-detect WebSocket URL based on current location
            function getWebSocketURL() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const hostname = window.location.hostname;
                const port = window.location.port ? ':' + window.location.port : '';
                
                return `${protocol}//${hostname}${port}/ws/${clientId}`;
            }
            
            function updateStatus(message, className) {
                const statusElement = document.getElementById('status');
                statusElement.textContent = message;
                statusElement.className = className;
            }
            
            function updateAudioLevel(rms) {
                const audioLevelElement = document.getElementById('audioLevel');
                audioLevelElement.textContent = `Audio Level: ${rms.toFixed(4)}`;
                
                // Change background color based on audio level
                if (rms > 0.01) {
                    audioLevelElement.style.backgroundColor = '#c8e6c9';
                } else if (rms > 0.001) {
                    audioLevelElement.style.backgroundColor = '#fff3e0';
                } else {
                    audioLevelElement.style.backgroundColor = '#f0f0f0';
                }
            }
            
            async function startRecording() {
                try {
                    // Request microphone access with enhanced constraints
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true,
                            volume: 1.0
                        }
                    });
                    
                    updateStatus('Status: Connecting...', 'recording');
                    
                    // Connect WebSocket
                    const wsUrl = getWebSocketURL();
                    console.log('Connecting to:', wsUrl);
                    websocket = new WebSocket(wsUrl);
                    
                    websocket.onopen = function(event) {
                        console.log('WebSocket connected');
                        updateStatus('Status: Connected and Recording', 'connected');
                        isRecording = true;
                    };
                    
                    websocket.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'transcription') {
                            const transcriptionDiv = document.getElementById('transcription');
                            const timestamp = new Date(data.timestamp).toLocaleTimeString();
                            
                            // Clear previous content and add new transcription
                            transcriptionDiv.innerHTML = '';
                            const transcriptionP = document.createElement('p');
                            transcriptionP.innerHTML = `<strong>[${timestamp}]:</strong> ${data.text}`;
                            transcriptionDiv.appendChild(transcriptionP);
                            
                            console.log('Transcription received:', data.text);
                        }
                    };
                    
                    websocket.onerror = function(error) {
                        console.error('WebSocket error:', error);
                        updateStatus('Status: Connection Error', 'error');
                    };
                    
                    websocket.onclose = function(event) {
                        console.log('WebSocket closed:', event.code, event.reason);
                        updateStatus('Status: Disconnected', 'stopped');
                        isRecording = false;
                    };
                    
                    // Setup Web Audio API with AudioWorklet
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 16000
                    });
                    
                    // Create AudioWorklet
                    const blob = new Blob([audioWorkletCode], { type: 'application/javascript' });
                    const workletUrl = URL.createObjectURL(blob);
                    
                    await audioContext.audioWorklet.addModule(workletUrl);
                    
                    const source = audioContext.createMediaStreamSource(stream);
                    audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
                    
                    audioWorkletNode.port.onmessage = function(event) {
                        if (event.data.type === 'audio' && websocket.readyState === WebSocket.OPEN) {
                            const { data: audioData, rms, counter } = event.data;
                            
                            // Update audio level display
                            updateAudioLevel(rms);
                            
                            // Debug logging
                            audioChunkCounter++;
                            if (audioChunkCounter % 100 === 0) {
                                console.log(`Sent ${audioChunkCounter} audio chunks, RMS: ${rms.toFixed(4)}`);
                            }
                            
                            // Convert to base64
                            const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
                            
                            websocket.send(JSON.stringify({
                                type: 'audio',
                                data: base64
                            }));
                        }
                    };
                    
                    source.connect(audioWorkletNode);
                    audioWorkletNode.connect(audioContext.destination);
                    
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    
                } catch (error) {
                    console.error('Error starting recording:', error);
                    updateStatus('Status: Error - ' + error.message, 'error');
                    alert('Error accessing microphone. Please check permissions and try again.');
                }
            }
            
            function stopRecording() {
                isRecording = false;
                
                if (audioWorkletNode) {
                    audioWorkletNode.disconnect();
                    audioWorkletNode = null;
                }
                
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
                
                if (websocket) {
                    websocket.close();
                }
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                updateStatus('Status: Stopped', 'stopped');
                updateAudioLevel(0);
                
                // Reset counters
                audioChunkCounter = 0;
            }
            
            // Handle page unload
            window.addEventListener('beforeunload', function(e) {
                if (isRecording) {
                    stopRecording();
                }
            });
        </script>
    </body>
    </html>
    """)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    # Start continuous transcription task
    transcription_task = asyncio.create_task(continuous_transcription(client_id))
    manager.transcription_tasks[client_id] = transcription_task
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio":
                audio_data = message.get("data")
                if audio_data:
                    # Process audio chunk
                    audio_array = process_audio_chunk(audio_data)
                    if len(audio_array) > 0:
                        manager.add_audio_chunk(client_id, audio_array)
                        
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        manager.disconnect(client_id)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "models_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
