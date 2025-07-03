# import whisper
# import sounddevice as sd
# import numpy as np
# import queue
# import threading
# import time
# from datetime import datetime
# from dotenv import load_dotenv
# from openai import OpenAI
# from APIKeyManager import load_api_key

# # Audio configuration
# SAMPLERATE_MIC = 16000
# SAMPLERATE_SYSTEM = 44100
# BLOCKSIZE = 4000
# PAUSE_SECONDS = 4
# CHUNK_TIME_SECONDS = 50

# # Global state
# model = whisper.load_model("base")
# client = None

# # Audio queues for different sources
# mic_audio_q = queue.Queue()
# system_audio_q = queue.Queue()

# # Shared buffer state
# buffer = []
# last_caption_time = time.time()
# last_summary_time = time.time()

# class AudioDeviceManager:
#     """Manages audio device detection and configuration"""
    
#     def __init__(self):
#         self.devices = None
#         self.system_device_id = None
#         self.system_supported_rates = []
#         self.refresh_devices()
    
#     def refresh_devices(self):
#         """Refresh the list of available audio devices"""
#         self.devices = sd.query_devices()
#         print("🔍 Available audio devices:")
#         print(f"{'ID':<4} {'Name':<50} {'In':<3} {'Out':<3} {'SR':<6}")
#         print("-" * 70)
        
#         for i, device in enumerate(self.devices):
#             name = device['name'][:47] + "..." if len(device['name']) > 50 else device['name']
#             print(f"{i:<4} {name:<50} {device['max_input_channels']:<3} {device['max_output_channels']:<3} {int(device['default_samplerate']):<6}")
    
#     def get_windows_defaults(self):
#         """Get Windows default input/output devices"""
#         try:
#             default_input = sd.default.device[0]
#             default_output = sd.default.device[1]
#             input_name = self.devices[default_input]['name']
#             output_name = self.devices[default_output]['name']
            
#             print(f"🎯 Windows defaults:")
#             print(f"   Input: Device {default_input} ({input_name})")
#             print(f"   Output: Device {default_output} ({output_name})")
            
#             return default_input, default_output
#         except Exception as e:
#             print(f"❌ Could not get default devices: {e}")
#             return None, None
    
#     def test_device_sample_rates(self, device_id):
#         """Test what sample rates a device supports"""
#         common_rates = [8000, 11025, 16000, 22050, 44100, 48000, 96000]
#         supported_rates = []
        
#         for rate in common_rates:
#             try:
#                 test_stream = sd.InputStream(
#                     device=device_id,
#                     channels=1,
#                     samplerate=rate,
#                     blocksize=1024,
#                     callback=lambda *args: None
#                 )
#                 test_stream.start()
#                 test_stream.stop()
#                 test_stream.close()
#                 supported_rates.append(rate)
#             except Exception:
#                 continue
        
#         return supported_rates
    
#     def find_system_audio_device(self):
#         """Find the best system audio capture device"""
#         print("🔍 Looking for system audio capture device...")
        
#         # Get Windows default output device
#         _, default_output = self.get_windows_defaults()
#         if default_output is not None:
#             default_name = self.devices[default_output]['name']
#             print(f"   Target: {default_name}")
        
#         # Look for loopback devices with priority
#         candidates = []
        
#         for i, device in enumerate(self.devices):
#             if device['max_input_channels'] > 0:
#                 name_lower = device['name'].lower()
#                 priority = 0
                
#                 # Priority scoring
#                 if 'stereo mix' in name_lower:
#                     priority = 10
#                 elif 'what u hear' in name_lower:
#                     priority = 9
#                 elif 'wasapi' in name_lower and 'loopback' in name_lower:
#                     priority = 8
#                 elif 'loopback' in name_lower:
#                     priority = 7
#                 elif 'monitor' in name_lower:
#                     priority = 6
                
#                 if priority > 0:
#                     supported_rates = self.test_device_sample_rates(i)
#                     candidates.append((priority, i, device['name'], supported_rates))
        
#         # Sort by priority
#         candidates.sort(reverse=True)
        
#         if candidates:
#             print("   Found system audio devices:")
#             for priority, device_id, name, rates in candidates:
#                 print(f"     Device {device_id}: {name}")
#                 print(f"       Priority: {priority}, Rates: {rates}")
                
#                 # Auto-select best device
#                 if priority == candidates[0][0] and (44100 in rates or 48000 in rates):
#                     self.system_device_id = device_id
#                     self.system_supported_rates = rates
#                     print(f"✅ Auto-selected: Device {device_id}")
#                     return device_id, rates
        
#         print("❌ No suitable system audio device found")
#         return None, []
    
#     def test_system_audio_levels(self, device_id, supported_rates):
#         """Test if system audio device is working"""
#         print(f"🧪 Testing system audio Device {device_id}...")
#         print("   ⚠️  PLAY SOME AUDIO NOW for this test!")
        
#         test_rate = 44100 if 44100 in supported_rates else supported_rates[0]
#         test_data = []
        
#         def test_callback(indata, frames, time, status):
#             if status:
#                 print(f"   Status: {status}")
#             test_data.append(indata.copy())
            
#             # Show live levels
#             volume = np.sqrt(np.mean(indata**2))
#             volume_db = 20 * np.log10(volume + 1e-10)
#             print(f"   Live: {volume:.6f} ({volume_db:.1f} dB)    ", end='\r')
        
#         try:
#             with sd.InputStream(
#                 device=device_id,
#                 channels=2,
#                 samplerate=test_rate,
#                 callback=test_callback,
#                 blocksize=1024
#             ):
#                 time.sleep(3)  # Test for 3 seconds
            
#             if test_data:
#                 combined_data = np.concatenate(test_data)
#                 max_level = np.max(np.abs(combined_data))
#                 avg_level = np.mean(np.abs(combined_data))
                
#                 print(f"\n   Results: Max={max_level:.6f}, Avg={avg_level:.6f}")
                
#                 if max_level > 0.01:
#                     print("   ✅ SUCCESS: System audio is working!")
#                     return True
#                 else:
#                     print("   ❌ PROBLEM: Audio levels too low")
#                     print("      Possible issues:")
#                     print("      - No audio playing")
#                     print("      - Audio going to different device")
#                     print("      - Stereo Mix disabled in Windows")
#                     return False
#         except Exception as e:
#             print(f"   ❌ Error testing device: {e}")
#             return False

# class DualAudioCapture:
#     """Handles capture from both microphone and system audio"""
    
#     def __init__(self):
#         self.device_manager = AudioDeviceManager()
#         self.mic_stream = None
#         self.system_stream = None
#         self.is_recording = False
#         self.lock = threading.Lock()
        
#     def mic_callback(self, indata, frames, time_info, status):
#         """Callback for microphone audio"""
#         if status:
#             print(f"🎤 Mic status: {status}")
        
#         if self.is_recording:
#             try:
#                 mic_audio_q.put(('mic', indata.copy()))
                
#                 # Debug: show mic levels (less frequent to avoid spam)
#                 if hasattr(self, '_mic_debug_counter'):
#                     self._mic_debug_counter += 1
#                 else:
#                     self._mic_debug_counter = 1
                    
#                 if self._mic_debug_counter % 50 == 0:  # Every 50th callback
#                     volume = np.sqrt(np.mean(indata**2))
#                     volume_db = 20 * np.log10(volume + 1e-10)
#                     print(f"🎤 Mic: {volume:.4f} ({volume_db:.1f} dB)")
                    
#             except Exception as e:
#                 print(f"❌ Error in mic callback: {e}")
    
#     def system_callback(self, indata, frames, time_info, status):
#         """Callback for system audio"""
#         if status:
#             print(f"🔊 System status: {status}")
        
#         if self.is_recording:
#             try:
#                 # Convert stereo to mono and resample for Whisper
#                 if len(indata.shape) > 1:
#                     mono_data = np.mean(indata, axis=1)
#                 else:
#                     mono_data = indata
                
#                 # Resample from 44100 to 16000 (simple decimation)
#                 resampled = mono_data[::int(SAMPLERATE_SYSTEM/SAMPLERATE_MIC)]
                
#                 system_audio_q.put(('system', resampled.copy()))
                
#                 # Debug: show system levels (less frequent to avoid spam)
#                 if hasattr(self, '_sys_debug_counter'):
#                     self._sys_debug_counter += 1
#                 else:
#                     self._sys_debug_counter = 1
                    
#                 if self._sys_debug_counter % 50 == 0:  # Every 50th callback
#                     volume = np.sqrt(np.mean(indata**2))
#                     volume_db = 20 * np.log10(volume + 1e-10)
#                     print(f"🔊 System: {volume:.4f} ({volume_db:.1f} dB)")
                    
#             except Exception as e:
#                 print(f"❌ Error in system callback: {e}")
    
#     def setup_devices(self):
#         """Setup and test audio devices"""
#         print("🚀 Setting up audio devices...")
        
#         # Find system audio device
#         system_device, supported_rates = self.device_manager.find_system_audio_device()
#         # Assume this passes for now.
#         print("✅ System audio setup complete")
#         return system_device, supported_rates
#         # if system_device is not None:
#         #     # Test system audio
#         #     if self.device_manager.test_system_audio_levels(system_device, supported_rates):
#         #         print("✅ System audio setup complete")
#         #         return system_device, supported_rates
#         #     else:
#         #         print("⚠️  System audio test failed - will continue with microphone only")
#         #         return None, []
        
#         # return None, []
    
#     def start_capture(self):
#         """Start capturing from both audio sources"""
#         print("🎧 Starting audio capture...")
        
#         # Setup devices
#         system_device, supported_rates = self.setup_devices()
        
#         self.is_recording = True
        
#         try:
#             # Start microphone stream
#             print("🎤 Starting microphone stream...")
#             self.mic_stream = sd.InputStream(
#                 channels=1,
#                 samplerate=SAMPLERATE_MIC,
#                 callback=self.mic_callback,
#                 blocksize=BLOCKSIZE
#             )
#             self.mic_stream.start()
#             print("✅ Microphone stream started")
            
#             # Debug: Check if mic stream is actually active
#             time.sleep(0.1)  # Give it a moment
#             print(f"🎤 Mic stream active: {self.mic_stream.active}")
            
#             # Start system audio stream if available
#             if system_device is not None:
#                 print(f"🔊 Starting system audio stream (Device {system_device})...")
#                 system_rate = 44100 if 44100 in supported_rates else supported_rates[0]
                
#                 try:
#                     self.system_stream = sd.InputStream(
#                         device=system_device,
#                         channels=2,
#                         samplerate=system_rate,
#                         callback=self.system_callback,
#                         blocksize=int(BLOCKSIZE * system_rate / SAMPLERATE_MIC)
#                     )
#                     self.system_stream.start()
#                     print("✅ System audio stream started")
                    
#                     # Debug: Check if system stream is actually active
#                     time.sleep(0.1)
#                     print(f"🔊 System stream active: {self.system_stream.active}")
                    
#                 except Exception as sys_error:
#                     print(f"❌ Failed to start system audio stream: {sys_error}")
#                     print(f"   Device: {system_device}")
#                     print(f"   Rate: {system_rate}")
#                     print(f"   Supported rates: {supported_rates}")
#                     self.system_stream = None
                    
#             else:
#                 print("⚠️  No system audio - microphone only")
                
#         except Exception as e:
#             print(f"❌ Error starting audio capture: {e}")
#             self.stop_capture()
#             raise
    
#     def stop_capture(self):
#         """Stop all audio capture"""
#         print("🛑 Stopping audio capture...")
#         self.is_recording = False
        
#         if self.mic_stream:
#             self.mic_stream.stop()
#             self.mic_stream.close()
#             self.mic_stream = None
        
#         if self.system_stream:
#             self.system_stream.stop()
#             self.system_stream.close()
#             self.system_stream = None
        
#         print("✅ Audio capture stopped")

# def summarize_buffer():
#     """Summarize the current buffer content"""
#     global buffer, last_summary_time
#     if not buffer:
#         return None

#     full_text = "\n".join(buffer).strip()
#     if not full_text:
#         return None

#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{
#                 "role": "user",
#                 "content": f"Summarize the following meeting transcript chunk:\n\n{full_text}"
#             }],
#             temperature=0.3
#         )
#         summary = response.choices[0].message.content
#         buffer.clear()
#         last_summary_time = time.time()
#         return summary
#     except Exception as e:
#         print(f"❌ Error creating summary: {e}")
#         return None

# def process_audio_queue(audio_queue, source_name, transcript_callback):
#     """Process audio from a specific queue with validation"""
#     temp_buffer = []
#     consecutive_errors = 0
    
#     while True:
#         try:
#             source, data = audio_queue.get(timeout=0.1)
            
#             # Validate audio data before adding to buffer
#             if data is None or len(data) == 0:
#                 consecutive_errors += 1
#                 if consecutive_errors % 10 == 0:  # Only log every 10th error
#                     print(f"⚠️  {source_name}: {consecutive_errors} consecutive empty chunks")
#                 continue
            
#             # Check for valid audio data
#             if np.isnan(data).any() or np.isinf(data).any():
#                 consecutive_errors += 1
#                 if consecutive_errors % 10 == 0:
#                     print(f"⚠️  {source_name}: {consecutive_errors} consecutive corrupted chunks")
#                 continue
            
#             # Reset error counter on good data
#             consecutive_errors = 0
#             temp_buffer.append(data)
            
#             # Process when we have enough data - INCREASED from 10 to 15 for Qt stability
#             if len(temp_buffer) >= 15:
#                 try:
#                     audio = np.concatenate(temp_buffer).flatten().astype(np.float32)
                    
#                     # Final validation before Whisper
#                     if len(audio) == 0:
#                         print(f"⚠️  {source_name}: Empty audio buffer after concatenation")
#                         temp_buffer.clear()
#                         continue
                    
#                     # RELAXED: Check audio length - more lenient for Qt
#                     if len(audio) < 800:  # Reduced from 1600 to 800
#                         print(f"⚠️  {source_name}: Audio too short ({len(audio)} samples)")
#                         temp_buffer.clear()
#                         continue
                    
#                     # Check if audio is just silence - more lenient threshold
#                     audio_level = np.sqrt(np.mean(audio**2))
#                     if audio_level < 5e-7:  # Reduced from 1e-6 to 5e-7
#                         if consecutive_errors % 20 == 0:  # Less frequent silence logging
#                             print(f"🔇 {source_name}: Skipping silence (level: {audio_level:.2e})")
#                         temp_buffer.clear()
#                         continue
                    
#                     # Normalize audio to prevent clipping
#                     max_val = np.max(np.abs(audio))
#                     if max_val > 0:
#                         audio = audio / max_val * 0.95
                    
#                     # PADDING: Add small amount of silence to help Whisper
#                     silence_padding = np.zeros(int(0.1 * 16000))  # 0.1 seconds
#                     audio = np.concatenate([silence_padding, audio, silence_padding])
                    
#                     print(f"🎵 {source_name}: Processing {len(audio)} samples, level: {audio_level:.6f}")
                    
#                     # Transcribe with error handling
#                     result = model.transcribe(audio, language="en", fp16=False)
#                     text = result.get("text", "").strip()
                    
#                     if text:
#                         timestamp = datetime.now().strftime("%H:%M:%S")
#                         line = f"[{timestamp}] {source_name}: {text}"
#                         print(f"📝 {line}")
#                         transcript_callback.emit(line)
                        
#                         global buffer, last_caption_time
#                         buffer.append(line)
#                         last_caption_time = time.time()
#                     else:
#                         print(f"🤐 {source_name}: No text detected")
                
#                 except Exception as whisper_error:
#                     print(f"❌ {source_name} Whisper error: {whisper_error}")
#                     if 'audio' in locals():
#                         print(f"   Audio shape: {audio.shape}")
#                         print(f"   Audio stats: min={np.min(audio):.6f}, max={np.max(audio):.6f}, mean={np.mean(audio):.6f}")
                
#                 temp_buffer.clear()
                
#         except queue.Empty:
#             continue
#         except Exception as e:
#             print(f"❌ Error processing {source_name} audio: {e}")
#             # Clear buffer on error to prevent cascading issues
#             temp_buffer.clear()
#             consecutive_errors += 1
#             continue

# def transcribe_dual_stream(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
#     """Main transcription loop handling both audio sources"""
#     global last_caption_time, last_summary_time
    
#     print("🎧 Starting dual-source transcription...")
    
#     # Start processing threads for each audio source
#     mic_thread = threading.Thread(
#         target=process_audio_queue,
#         args=(mic_audio_q, "MIC", transcript_callback),
#         daemon=True
#     )
    
#     system_thread = threading.Thread(
#         target=process_audio_queue,
#         args=(system_audio_q, "SYSTEM", transcript_callback),
#         daemon=True
#     )
    
#     mic_thread.start()
#     system_thread.start()
    
#     # Main loop for summaries and monitoring
#     while should_continue():
#         try:
#             time.sleep(1)  # Check every second
#             now = time.time()
            
#             # Check if we should create a summary
#             if is_auto_summary_enabled() and buffer and (
#                 now - last_caption_time >= PAUSE_SECONDS or
#                 now - last_summary_time >= get_summary_interval_sec()
#             ):
#                 print("📋 Creating summary...")
#                 summary = summarize_buffer()
#                 if summary:
#                     summary_callback.emit(summary)
                    
#         except Exception as e:
#             print(f"❌ Error in transcription loop: {e}")
#             continue

# def run_transcription(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
#     """Main entry point for transcription system"""
#     global client
    
#     print("🚀 Initializing transcription system...")
    
#     # Setup OpenAI client
#     api_key = load_api_key()
#     if not api_key:
#         raise RuntimeError("OpenAI API key is not set. Please go to Settings and add your key.")
#     client = OpenAI(api_key=api_key)
    
#     # Setup dual audio capture
#     audio_capture = DualAudioCapture()
    
#     try:
#         # Start audio capture
#         audio_capture.start_capture()
        
#         # Start transcription
#         transcribe_dual_stream(
#             transcript_callback, 
#             summary_callback, 
#             should_continue, 
#             is_auto_summary_enabled, 
#             get_summary_interval_sec
#         )
        
#     except Exception as e:
#         print(f"❌ Transcription error: {e}")
#         raise
#     finally:
#         audio_capture.stop_capture()
#         print("🏁 Transcription system stopped")

# # Legacy function for backward compatibility
# def run_transcription_legacy(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
#     """Legacy single-source transcription (microphone only)"""
#     global client
    
#     print("🎧 Running legacy single-source transcription...")
    
#     api_key = load_api_key()
#     if not api_key:
#         raise RuntimeError("OpenAI API key is not set. Please go to Settings and add your key.")
#     client = OpenAI(api_key=api_key)

#     def audio_callback(indata, frames, time_info, status):
#         if status:
#             print("⚠️", status)
#         mic_audio_q.put(('mic', indata.copy()))

#     with sd.InputStream(channels=1, samplerate=SAMPLERATE_MIC,
#                         callback=audio_callback, blocksize=BLOCKSIZE):
#         process_audio_queue(mic_audio_q, "MIC", transcript_callback)

# enhanced_transcription_worker.py
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from APIKeyManager import load_api_key

# Audio configuration
SAMPLERATE_MIC = 16000
SAMPLERATE_SYSTEM = 44100
BLOCKSIZE = 4000
PAUSE_SECONDS = 4
CHUNK_TIME_SECONDS = 50

# Global state
model = whisper.load_model("base")
client = None

# Audio queues for different sources
mic_audio_q = queue.Queue()
system_audio_q = queue.Queue()

# Shared buffer state
buffer = []
last_caption_time = time.time()
last_summary_time = time.time()

class AudioDeviceManager:
    """Manages audio device detection and configuration"""
    
    def __init__(self):
        self.devices = None
        self.system_device_id = None
        self.system_supported_rates = []
        self.refresh_devices()
    
    def refresh_devices(self):
        """Refresh the list of available audio devices"""
        self.devices = sd.query_devices()
        print("🔍 Available audio devices:")
        print(f"{'ID':<4} {'Name':<50} {'In':<3} {'Out':<3} {'SR':<6}")
        print("-" * 70)
        
        for i, device in enumerate(self.devices):
            name = device['name'][:47] + "..." if len(device['name']) > 50 else device['name']
            print(f"{i:<4} {name:<50} {device['max_input_channels']:<3} {device['max_output_channels']:<3} {int(device['default_samplerate']):<6}")
    
    def get_windows_defaults(self):
        """Get Windows default input/output devices"""
        try:
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            input_name = self.devices[default_input]['name']
            output_name = self.devices[default_output]['name']
            
            print(f"🎯 Windows defaults:")
            print(f"   Input: Device {default_input} ({input_name})")
            print(f"   Output: Device {default_output} ({output_name})")
            
            return default_input, default_output
        except Exception as e:
            print(f"❌ Could not get default devices: {e}")
            return None, None
    
    def test_device_sample_rates(self, device_id):
        """Test what sample rates a device supports"""
        common_rates = [8000, 11025, 16000, 22050, 44100, 48000, 96000]
        supported_rates = []
        
        for rate in common_rates:
            try:
                test_stream = sd.InputStream(
                    device=device_id,
                    channels=1,
                    samplerate=rate,
                    blocksize=1024,
                    callback=lambda *args: None
                )
                test_stream.start()
                test_stream.stop()
                test_stream.close()
                supported_rates.append(rate)
            except Exception:
                continue
        
        return supported_rates
    
    def find_system_audio_device(self):
        """Find the best system audio capture device"""
        print("🔍 Looking for system audio capture device...")
        
        # Get Windows default output device
        _, default_output = self.get_windows_defaults()
        if default_output is not None:
            default_name = self.devices[default_output]['name']
            print(f"   Target: {default_name}")
        
        # Look for loopback devices with priority
        candidates = []
        
        for i, device in enumerate(self.devices):
            if device['max_input_channels'] > 0:
                name_lower = device['name'].lower()
                priority = 0
                
                # Priority scoring
                if 'stereo mix' in name_lower:
                    priority = 10
                elif 'what u hear' in name_lower:
                    priority = 9
                elif 'wasapi' in name_lower and 'loopback' in name_lower:
                    priority = 8
                elif 'loopback' in name_lower:
                    priority = 7
                elif 'monitor' in name_lower:
                    priority = 6
                
                if priority > 0:
                    supported_rates = self.test_device_sample_rates(i)
                    candidates.append((priority, i, device['name'], supported_rates))
        
        # Sort by priority
        candidates.sort(reverse=True)
        
        if candidates:
            print("   Found system audio devices:")
            for priority, device_id, name, rates in candidates:
                print(f"     Device {device_id}: {name}")
                print(f"       Priority: {priority}, Rates: {rates}")
                
                # Auto-select best device
                if priority == candidates[0][0] and (44100 in rates or 48000 in rates):
                    self.system_device_id = device_id
                    self.system_supported_rates = rates
                    print(f"✅ Auto-selected: Device {device_id}")
                    return device_id, rates
        
        print("❌ No suitable system audio device found")
        return None, []
    
    def test_system_audio_levels(self, device_id, supported_rates):
        """Test if system audio device is working"""
        print(f"🧪 Testing system audio Device {device_id}...")
        print("   ⚠️  PLAY SOME AUDIO NOW for this test!")
        
        test_rate = 44100 if 44100 in supported_rates else supported_rates[0]
        test_data = []
        
        def test_callback(indata, frames, time, status):
            if status:
                print(f"   Status: {status}")
            test_data.append(indata.copy())
            
            # Show live levels
            volume = np.sqrt(np.mean(indata**2))
            volume_db = 20 * np.log10(volume + 1e-10)
            print(f"   Live: {volume:.6f} ({volume_db:.1f} dB)    ", end='\r')
        
        try:
            with sd.InputStream(
                device=device_id,
                channels=2,
                samplerate=test_rate,
                callback=test_callback,
                blocksize=1024
            ):
                time.sleep(3)  # Test for 3 seconds
            
            if test_data:
                combined_data = np.concatenate(test_data)
                max_level = np.max(np.abs(combined_data))
                avg_level = np.mean(np.abs(combined_data))
                
                print(f"\n   Results: Max={max_level:.6f}, Avg={avg_level:.6f}")
                
                if max_level > 0.01:
                    print("   ✅ SUCCESS: System audio is working!")
                    return True
                else:
                    print("   ❌ PROBLEM: Audio levels too low")
                    print("      Possible issues:")
                    print("      - No audio playing")
                    print("      - Audio going to different device")
                    print("      - Stereo Mix disabled in Windows")
                    return False
        except Exception as e:
            print(f"   ❌ Error testing device: {e}")
            return False

class DualAudioCapture:
    """Handles capture from both microphone and system audio"""
    
    def __init__(self):
        self.device_manager = AudioDeviceManager()
        self.mic_stream = None
        self.system_stream = None
        self.is_recording = False
        self.lock = threading.Lock()
        
    def mic_callback(self, indata, frames, time_info, status):
        """Callback for microphone audio"""
        if status:
            print(f"🎤 Mic status: {status}")
        
        if self.is_recording:
            try:
                mic_audio_q.put(('mic', indata.copy()))
                
                # Debug: show mic levels (less frequent to avoid spam)
                if hasattr(self, '_mic_debug_counter'):
                    self._mic_debug_counter += 1
                else:
                    self._mic_debug_counter = 1
                    
                if self._mic_debug_counter % 50 == 0:  # Every 50th callback
                    volume = np.sqrt(np.mean(indata**2))
                    volume_db = 20 * np.log10(volume + 1e-10)
                    print(f"🎤 Mic: {volume:.4f} ({volume_db:.1f} dB)")
                    
            except Exception as e:
                print(f"❌ Error in mic callback: {e}")
    
    def system_callback(self, indata, frames, time_info, status):
        """Callback for system audio"""
        if status:
            print(f"🔊 System status: {status}")
        
        if self.is_recording:
            try:
                # Convert stereo to mono and resample for Whisper
                if len(indata.shape) > 1:
                    mono_data = np.mean(indata, axis=1)
                else:
                    mono_data = indata
                
                # Resample from 44100 to 16000 (simple decimation)
                resampled = mono_data[::int(SAMPLERATE_SYSTEM/SAMPLERATE_MIC)]
                
                system_audio_q.put(('system', resampled.copy()))
                
                # Debug: show system levels (less frequent to avoid spam)
                if hasattr(self, '_sys_debug_counter'):
                    self._sys_debug_counter += 1
                else:
                    self._sys_debug_counter = 1
                    
                if self._sys_debug_counter % 50 == 0:  # Every 50th callback
                    volume = np.sqrt(np.mean(indata**2))
                    volume_db = 20 * np.log10(volume + 1e-10)
                    print(f"🔊 System: {volume:.4f} ({volume_db:.1f} dB)")
                    
            except Exception as e:
                print(f"❌ Error in system callback: {e}")
    
    def setup_devices(self):
        """Setup and test audio devices"""
        print("🚀 Setting up audio devices...")
        
        # Find system audio device
        system_device, supported_rates = self.device_manager.find_system_audio_device()
        # Assume this passes for now.
        print("✅ System audio setup complete")
        return system_device, supported_rates
        # if system_device is not None:
        #     # Test system audio
        #     if self.device_manager.test_system_audio_levels(system_device, supported_rates):
        #         print("✅ System audio setup complete")
        #         return system_device, supported_rates
        #     else:
        #         print("⚠️  System audio test failed - will continue with microphone only")
        #         return None, []
        
        # return None, []
    
    def start_capture(self):
        """Start capturing from both audio sources"""
        print("🎧 Starting audio capture...")
        
        # Setup devices
        system_device, supported_rates = self.setup_devices()
        
        self.is_recording = True
        
        try:
            # Start microphone stream
            print("🎤 Starting microphone stream...")
            self.mic_stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLERATE_MIC,
                callback=self.mic_callback,
                blocksize=BLOCKSIZE
            )
            self.mic_stream.start()
            print("✅ Microphone stream started")
            
            # Debug: Check if mic stream is actually active
            time.sleep(0.1)  # Give it a moment
            print(f"🎤 Mic stream active: {self.mic_stream.active}")
            
            # Start system audio stream if available
            if system_device is not None:
                print(f"🔊 Starting system audio stream (Device {system_device})...")
                system_rate = 44100 if 44100 in supported_rates else supported_rates[0]
                
                try:
                    self.system_stream = sd.InputStream(
                        device=system_device,
                        channels=2,
                        samplerate=system_rate,
                        callback=self.system_callback,
                        blocksize=int(BLOCKSIZE * system_rate / SAMPLERATE_MIC)
                    )
                    self.system_stream.start()
                    print("✅ System audio stream started")
                    
                    # Debug: Check if system stream is actually active
                    time.sleep(0.1)
                    print(f"🔊 System stream active: {self.system_stream.active}")
                    
                except Exception as sys_error:
                    print(f"❌ Failed to start system audio stream: {sys_error}")
                    print(f"   Device: {system_device}")
                    print(f"   Rate: {system_rate}")
                    print(f"   Supported rates: {supported_rates}")
                    self.system_stream = None
                    
            else:
                print("⚠️  No system audio - microphone only")
                
        except Exception as e:
            print(f"❌ Error starting audio capture: {e}")
            self.stop_capture()
            raise
    
    def stop_capture(self):
        """Stop all audio capture"""
        print("🛑 Stopping audio capture...")
        self.is_recording = False
        
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None
        
        if self.system_stream:
            self.system_stream.stop()
            self.system_stream.close()
            self.system_stream = None
        
        print("✅ Audio capture stopped")

def summarize_buffer():
    """Summarize the current buffer content"""
    global buffer, last_summary_time
    if not buffer:
        return None

    full_text = "\n".join(buffer).strip()
    if not full_text:
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Summarize the following meeting transcript chunk:\n\n{full_text}"
            }],
            temperature=0.3
        )
        summary = response.choices[0].message.content
        buffer.clear()
        last_summary_time = time.time()
        return summary
    except Exception as e:
        print(f"❌ Error creating summary: {e}")
        return None

def process_audio_queue(audio_queue, source_name, transcript_callback):
    """Process audio from a specific queue with validation"""
    temp_buffer = []
    
    while True:
        try:
            source, data = audio_queue.get(timeout=0.1)
            
            # Validate audio data before adding to buffer
            if data is None or len(data) == 0:
                print(f"⚠️  {source_name}: Skipping empty audio chunk")
                continue
            
            # Check for valid audio data
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"⚠️  {source_name}: Skipping corrupted audio chunk (NaN/Inf)")
                continue
                
            temp_buffer.append(data)
            
            # Process when we have enough data
            if len(temp_buffer) >= 10:
                try:
                    audio = np.concatenate(temp_buffer).flatten().astype(np.float32)
                    
                    # Final validation before Whisper
                    if len(audio) == 0:
                        print(f"⚠️  {source_name}: Empty audio buffer after concatenation")
                        temp_buffer.clear()
                        continue
                    
                    # Check audio length - Whisper needs at least some audio
                    if len(audio) < 1600:  # Less than 0.1 seconds at 16kHz
                        print(f"⚠️  {source_name}: Audio too short ({len(audio)} samples)")
                        temp_buffer.clear()
                        continue
                    
                    # Check if audio is just silence
                    audio_level = np.sqrt(np.mean(audio**2))
                    if audio_level < 1e-6:  # Very quiet threshold
                        print(f"🔇 {source_name}: Skipping silence (level: {audio_level:.2e})")
                        temp_buffer.clear()
                        continue
                    
                    # Normalize audio to prevent clipping
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio / max_val * 0.95  # Normalize to 95% to prevent clipping
                    
                    print(f"🎵 {source_name}: Processing {len(audio)} samples, level: {audio_level:.6f}")
                    
                    # Transcribe with error handling
                    result = model.transcribe(audio, language="en", fp16=False)
                    text = result.get("text", "").strip()
                    
                    if text:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        line = f"[{timestamp}] {source_name}: {text}"
                        print(f"📝 {line}")
                        transcript_callback.emit(line)
                        
                        global buffer, last_caption_time
                        buffer.append(line)
                        last_caption_time = time.time()
                    else:
                        print(f"🤐 {source_name}: No text detected")
                
                except Exception as whisper_error:
                    print(f"❌ {source_name} Whisper error: {whisper_error}")
                    print(f"   Audio shape: {audio.shape if 'audio' in locals() else 'undefined'}")
                    print(f"   Audio stats: min={np.min(audio):.6f}, max={np.max(audio):.6f}, mean={np.mean(audio):.6f}")
                
                temp_buffer.clear()
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Error processing {source_name} audio: {e}")
            # Clear buffer on error to prevent cascading issues
            temp_buffer.clear()
            continue

def transcribe_dual_stream(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
    """Main transcription loop handling both audio sources"""
    global last_caption_time, last_summary_time
    
    print("🎧 Starting dual-source transcription...")
    
    # Start processing threads for each audio source
    mic_thread = threading.Thread(
        target=process_audio_queue,
        args=(mic_audio_q, "MIC", transcript_callback),
        daemon=True
    )
    
    system_thread = threading.Thread(
        target=process_audio_queue,
        args=(system_audio_q, "SYSTEM", transcript_callback),
        daemon=True
    )
    
    mic_thread.start()
    system_thread.start()
    
    # Main loop for summaries and monitoring
    while should_continue():
        try:
            time.sleep(1)  # Check every second
            now = time.time()
            
            # Check if we should create a summary
            if is_auto_summary_enabled() and buffer and (
                now - last_caption_time >= PAUSE_SECONDS or
                now - last_summary_time >= get_summary_interval_sec()
            ):
                print("📋 Creating summary...")
                summary = summarize_buffer()
                if summary:
                    summary_callback.emit(summary)
                    
        except Exception as e:
            print(f"❌ Error in transcription loop: {e}")
            continue

def run_transcription(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
    """Main entry point for transcription system"""
    global client
    
    print("🚀 Initializing transcription system...")
    
    # Setup OpenAI client
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("OpenAI API key is not set. Please go to Settings and add your key.")
    client = OpenAI(api_key=api_key)
    
    # Setup dual audio capture
    audio_capture = DualAudioCapture()
    
    try:
        # Start audio capture
        audio_capture.start_capture()
        
        # Start transcription
        transcribe_dual_stream(
            transcript_callback, 
            summary_callback, 
            should_continue, 
            is_auto_summary_enabled, 
            get_summary_interval_sec
        )
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise
    finally:
        audio_capture.stop_capture()
        print("🏁 Transcription system stopped")

# Legacy function for backward compatibility
def run_transcription_legacy(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
    """Legacy single-source transcription (microphone only)"""
    global client
    
    print("🎧 Running legacy single-source transcription...")
    
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("OpenAI API key is not set. Please go to Settings and add your key.")
    client = OpenAI(api_key=api_key)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("⚠️", status)
        mic_audio_q.put(('mic', indata.copy()))

    with sd.InputStream(channels=1, samplerate=SAMPLERATE_MIC,
                        callback=audio_callback, blocksize=BLOCKSIZE):
        process_audio_queue(mic_audio_q, "MIC", transcript_callback)
