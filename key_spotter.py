import numpy as np
import librosa
import pyfiglet
from halo import Halo
from colorama import Fore, init
import time
import wave
import onnxruntime
from utils import log_specgram , MFCC , MSLFB
import tqdm


def predict_model(signal_resampled, SAMPLING_RATE):
    audio_length = 48000
    if len(signal_resampled.shape)==2:
      signal_resampled = np.mean(signal_resampled, axis=1)

    if len(signal_resampled) > audio_length:
      signal_resampled = signal_resampled[:audio_length]
    elif len(signal_resampled) != audio_length:
      zeros_needed = audio_length - len(signal_resampled)
      signal_resampled = np.append(signal_resampled, np.zeros((zeros_needed)))

    msLFB_spec = MSLFB(signal_resampled)
    mcff_spec = MFCC(signal_resampled)

    signal_resampled.fill(0)
    # Assume x_new_mfcc and x_new_mslfb are the new input data
    x_new_mfcc_reshaped_spec = np.reshape(mcff_spec, (1, mcff_spec.shape[0], mcff_spec.shape[1], 1)).astype(np.float32)
    x_new_mslfb_reshaped_spec = np.reshape(msLFB_spec, (1, msLFB_spec.shape[0], msLFB_spec.shape[1], 1)).astype(np.float32)

    # Make predictions on new data
    # Prepare the input dat

    # Make predictions on new data
    # Prepare the input data
    input_data = {'conv2d_18_input': x_new_mfcc_reshaped_spec, 'conv2d_21_input': x_new_mslfb_reshaped_spec}
    # Run the prediction
    try:
        y_probs = sess.run(None, input_data)
        arr = y_probs[0]
        selected_values = arr[arr > 0.00005]

        # Convert the output to class predictions
        if len(selected_values) == 0:
            return  None

        else:
            y_preds = selected_values.argmax()
            return classes[y_preds]
    except:
        print("[!Notice!] Tensor Received incorrect Value".format(Fore.BLUE, Fore.RESET))
        return None



def is_wav_file(file_path):
    try:
        with wave.open(file_path, 'r') as wf:
            # check if the file is a WAV file
            if wf.getnchannels() > 0:
                return True
            else:
                return False
    except wave.Error:
        return False
def show_painting():

    global path_of_file
    try:

        print(pyfiglet.figlet_format('Keyword Spotter', justify='center'))
        print("""\t{}______
                            Version 1.0
                            Created By : Jimma University , Electrical and Computer Enginnering                           
                                                    {}\n""".format(Fore.BLUE, Fore.RESET))
        time.sleep(5)
        print()

        path_of_file = input("""\t{}Please Give Me Path of the Audio File format {}\n""".format(Fore.GREEN, Fore.RESET))
        print("Given Data begin to processed!")

    except Exception as e:
        return "[-] Couldn't show the start painting: {}".format(e)



show_painting()

classes = ['Eserat', 'Agtew', 'Dferat', 'Tlefat', 'Reshinachew', 'Tsetargew',
           'Forjid' , 'Shibr' , 'Gejera' , 'Ets' , 'Gubo' , 'Zrefew' , 'Refrfew' ,
           'Dfaw' , 'Selilew' , 'Musina' , 'Zelzlew' , 'Afendaw' , 'Agayew', 'Zerirew' ,'Unknown' , 'Silence']

MODEL_PATH = "model/asb_ensemble_final.onnx"
sess = onnxruntime.InferenceSession(MODEL_PATH)
# set the desired audio parameters
RATE = 16000
CHUNK_SIZE =48000
HALF_CHUNK_SIZE = CHUNK_SIZE // 2


# read in the audio data and resample it
audio, sr = librosa.load('myaudio.wav', sr=RATE)

print(f"Loaded Audio {len(audio)}")

# divide the audio into chunks of 48000 samples
chunks = [audio[i:i + CHUNK_SIZE] for i in range(0, len(audio), CHUNK_SIZE)]

print(f"Length of {len(chunks)}")
print(f"Data of {chunks[5]}")




# create a new list of chunks by taking the last half of the first chunk and the first half of the second chunk
new_chunks = []
for i in range(len(chunks) - 1):
    new_chunk = np.concatenate((chunks[i][HALF_CHUNK_SIZE:], chunks[i + 1][:HALF_CHUNK_SIZE]))
    new_chunks.append(new_chunk)
print(f"Length of the new chunk is  {len(new_chunks)}")
print(f"Data of {new_chunks[5]}")



result1 = []
result2 = []
time = 0
for audio_data in tqdm.tqdm(chunks, desc="Processing audio chunks"):
    res = predict_model(audio_data, 16000)
    if res == None:
        continue
    result1.append(res)



time = 2
for audio_data in tqdm.tqdm(new_chunks, desc="Processing audio chunks"):
    res = predict_model(audio_data, 16000)
    if res == None:
        continue
    result2.append(res)


print()
print("==============================================")
print("""\t{}[*] We have detected {} Sequencial and {} Intermediatary Amharic criminal words in this audio file [*]{}\n""".format(Fore.GREEN, len(result1), len(result2), Fore.RESET))
for  disp in result1:
    if disp!= None:
        minute = int(time/60)
        second = time%60
        print(f'onDetect Sequencial:  {disp} -> around {minute}:{second}')
    time = time+3
for  disp in result2:
    if disp!= None:
        minute = int(time/60)
        second = time%60
        print(f'onDetect Intermediatary:  {disp} -> around {minute}:{second}')
    time = time+3

