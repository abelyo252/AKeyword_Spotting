import pyaudio
import wave

CHUNK = 1024  # Record in chunks of 1024 samples
FORMAT = pyaudio.paInt16  # 16 bits per sample
CHANNELS = 1
RATE = 44100  # Record at 44100 samples per second
SECONDS = 10
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=FORMAT,
                channels=CHANNELS ,
                rate=RATE,
                frames_per_buffer=CHUNK,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(RATE / CHUNK * SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('[*] Finished recording')



def get_audio(audio_data):
    audio_name = 'record.wav'
    wf = wave.open(audio_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_data))
    wf.close()
    print("[*] Audio Saved Successfully !")

get_audio(frames)