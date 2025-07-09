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
import soundfile as sf
from pysilero_vad import SileroVoiceActivityDetector
import queue
import threading
import time
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Real-Time STT API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
batch_size = 16
model = None
align_model = None
metadata = None
vad_detector = None

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, queue.Queue] = {}
        self.transcription_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = queue.Queue()
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.transcription_tasks:
            self.transcription_tasks[client_id].cancel()
            del self.transcription_tasks[client_id]
        logger.info(f"Client {client_id} disconnected")
        
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                
    def add_audio_chunk(self, client_id: str, audio_data: bytes):
        if client_id in self.audio_buffers:
            self.audio_buffers[client_id].put(audio_data)

manager = ConnectionManager()

# Initialize models
def initialize_models():
    global model, align_model, metadata, vad_detector
    
    try:
        # Initialize WhisperX model
        logger.info("Loading WhisperX model...")
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        
        # Initialize alignment model
        logger.info("Loading alignment model...")
        align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
        
        # Initialize Silero VAD
        logger.info("Loading Silero VAD...")
        vad_detector = SileroVoiceActivityDetector()
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

# Audio processing functions
def process_audio_chunk(audio_data: bytes) -> np.ndarray:
    """Convert audio bytes to numpy array for processing"""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Convert to numpy array
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
        return audio_array
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        return np.array([])

def detect_voice_activity(audio_array: np.ndarray) -> bool:
    """Detect voice activity using Silero VAD"""
    try:
        if len(audio_array) == 0:
            return False
            
        # Ensure audio is correct length for VAD (512 samples for 16kHz)
        chunk_size = vad_detector.chunk_samples()
        if len(audio_array) < chunk_size:
            # Pad with zeros
            audio_array = np.pad(audio_array, (0, chunk_size - len(audio_array)))
        elif len(audio_array) > chunk_size:
            # Take first chunk
            audio_array = audio_array[:chunk_size]
            
        # Convert to bytes
        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        
        # Detect voice activity
        vad_score = vad_detector(audio_bytes)
        return vad_score >= 0.5
        
    except Exception as e:
        logger.error(f"Error in VAD detection: {e}")
        return False

async def transcribe_audio(audio_array: np.ndarray) -> str:
    """Transcribe audio using WhisperX"""
    try:
        if len(audio_array) == 0:
            return ""
            
        # Transcribe with WhisperX
        result = model.transcribe(audio_array, batch_size=batch_size)
        
        # Align transcript
        if align_model is not None and result["segments"]:
            result = whisperx.align(result["segments"], align_model, metadata, 
                                 audio_array, device, return_char_alignments=False)
        
        # Extract text
        text = ""
        if "segments" in result:
            for segment in result["segments"]:
                if "text" in segment:
                    text += segment["text"]
                    
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return ""

# Background transcription task
async def continuous_transcription(client_id: str):
    """Continuously process audio chunks for transcription"""
    audio_buffer = []
    last_transcription_time = time.time()
    
    while client_id in manager.active_connections:
        try:
            # Get audio chunks from buffer
            audio_queue = manager.audio_buffers.get(client_id)
            if not audio_queue or audio_queue.empty():
                await asyncio.sleep(0.1)
                continue
                
            # Process available audio chunks
            while not audio_queue.empty():
                try:
                    audio_data = audio_queue.get_nowait()
                    audio_array = process_audio_chunk(audio_data)
                    
                    if len(audio_array) > 0:
                        # Check for voice activity
                        if detect_voice_activity(audio_array):
                            audio_buffer.extend(audio_array)
                            
                except queue.Empty:
                    break
                    
            # Transcribe if buffer has enough audio (1 second worth)
            if len(audio_buffer) >= 16000:  # 1 second at 16kHz
                current_time = time.time()
                
                # Transcribe every 2 seconds or when buffer is large
                if (current_time - last_transcription_time >= 2.0 or 
                    len(audio_buffer) >= 48000):  # 3 seconds
                    
                    # Convert to numpy array
                    audio_array = np.array(audio_buffer, dtype=np.float32)
                    
                    # Transcribe
                    text = await transcribe_audio(audio_array)
                    
                    if text:
                        # Send transcription result
                        await manager.send_personal_message({
                            "type": "transcription",
                            "text": text,
                            "timestamp": datetime.now().isoformat()
                        }, client_id)
                        
                    # Clear buffer and update time
                    audio_buffer = []
                    last_transcription_time = current_time
                    
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
            }
            #transcription {
                border: 1px solid #ccc;
                padding: 20px;
                min-height: 200px;
                margin: 20px 0;
                background-color: #f9f9f9;
            }
            button {
                padding: 10px 20px;
                margin: 10px;
                font-size: 16px;
                cursor: pointer;
            }
            #startBtn {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            #stopBtn {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
            }
            #status {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
            }
            .recording {
                background-color: #ffeb3b;
                color: #333;
            }
            .stopped {
                background-color: #e0e0e0;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>Real-Time Speech-to-Text</h1>
        <div id="status" class="stopped">Status: Stopped</div>
        <button id="startBtn" onclick="startRecording()">Start Recording</button>
        <button id="stopBtn" onclick="stopRecording()" disabled>Stop Recording</button>
        <div id="transcription"></div>
        
        <script>
            let mediaRecorder;
            let websocket;
            let clientId = 'client_' + Date.now();
            
            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        }
                    });
                    
                    // Connect WebSocket
                    websocket = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
                    
                    websocket.onopen = function(event) {
                        console.log('WebSocket connected');
                    };
                    
                    websocket.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'transcription') {
                            document.getElementById('transcription').innerHTML += 
                                `<p><strong>${data.timestamp}:</strong> ${data.text}</p>`;
                        }
                    };
                    
                    websocket.onerror = function(error) {
                        console.error('WebSocket error:', error);
                    };
                    
                    // Setup MediaRecorder
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=opus'
                    });
                    
                    mediaRecorder.ondataavailable = function(event) {
                        if (event.data.size > 0) {
                            const reader = new FileReader();
                            reader.onload = function() {
                                const arrayBuffer = reader.result;
                                const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                                if (websocket.readyState === WebSocket.OPEN) {
                                    websocket.send(JSON.stringify({
                                        type: 'audio',
                                        data: base64
                                    }));
                                }
                            };
                            reader.readAsArrayBuffer(event.data);
                        }
                    };
                    
                    mediaRecorder.start(1000); // Send data every 1 second
                    
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('status').className = 'recording';
                    document.getElementById('status').textContent = 'Status: Recording...';
                    
                } catch (error) {
                    console.error('Error starting recording:', error);
                    alert('Error accessing microphone. Please check permissions.');
                }
            }
            
            function stopRecording() {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                    mediaRecorder.stream.getTracks().forEach(track => track.stop());
                }
                
                if (websocket) {
                    websocket.close();
                }
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('status').className = 'stopped';
                document.getElementById('status').textContent = 'Status: Stopped';
            }
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
                    manager.add_audio_chunk(client_id, audio_data)
                    
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

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    initialize_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
