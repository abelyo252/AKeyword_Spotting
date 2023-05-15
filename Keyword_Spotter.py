"""
Keyword Spotting
By: Jimma University , Electrical and computer Enginnering Student
Dataset Collected : 2011 Batch Computer Stream Student
Website: https://github.com/abelyo252/AKeyword_Spotting
"""

# Import standard modules.
from math import sqrt
import random

# Import non-standard modules.
import pygame
import pyaudio
import numpy as np
import onnxruntime
from utils import log_specgram , MFCC , MSLFB
import wave
from termcolor import colored
from halo import Halo
from colorama import Fore, init


class Keyword_Spotter:
    """
    Hold all information that used for display whole
    """

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, MODEL_PATH):
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT
        self.model_path = MODEL_PATH

        # Initialise PyGame.
        pygame.init()
        self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('Keyword Spotting')
        self.background = pygame.image.load("asset/keyspot.png").convert()
        self.font = pygame.font.Font('Fonts/hoog0553.ttf', 22)

        # User Defined Variable
        self.model = self.get_model()

    # User Define Functions
    def get_model(self):
        sess = onnxruntime.InferenceSession(self.model_path)
        return sess

    def update(self, dt):
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def draw(self, data, result):
        """
        Draw things to the window. Called once per frame.
        """
        self.window.blit(self.background, (0, 0))
        fft_complex = np.fft.fft(data, n=1024)
        # fft_distance = np.zeros(len(fft_complex))

        s = 0
        max_val = sqrt(max(v.real * v.real + v.imag * v.imag for v in fft_complex))
        scale_value = self.SCREEN_HEIGHT / max_val
        for i, v in enumerate(fft_complex):
            # v = complex(v.real / dist1, v.imag / dist1)
            dist = sqrt(v.real * v.real + v.imag * v.imag)
            mapped_dist = dist * scale_value
            s += mapped_dist

            color = (random.randint(80, 90), random.randint(50, 60), random.randint(180, 195))
            pygame.draw.line(self.window, color, (i, self.SCREEN_HEIGHT), (i, self.SCREEN_HEIGHT - mapped_dist))
        # print(s / len(fft_complex))

        text = self.font.render(result, True, (255,255,255))
        textRect = text.get_rect()
        textRect.x = 450
        textRect.y = 482
        self.window.blit(text, textRect)



    def get_audio(self , audio_data):
        audio_name = 'record.wav'
        wf = wave.open(audio_name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(audio_data))
        wf.close()
        print("[*] Audio Saved Successfully !")



def main():
    """
    main file that run window
    """

    # Initialise Pyaudio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 1.5
    SAMPLING_RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                            channels = CHANNELS,
                            rate = RATE,
                            input = True,
                            frames_per_buffer = CHUNK)


    # start recording
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks() / 1000.0  # current time in seconds

    #############  Parameter Configure This System ###################
    ##################################################################

    result = ''
    label_names = ['Eserat', 'Agtew', 'Dferat', 'Tlefat', 'Reshinachew', 'Tsetargew',
                   'Forjid', 'Shibr', 'Gejera', 'Ets', 'Gubo', 'Zrefew', 'Refrfew',
                   'Dfaw', 'Selilew', 'Musina', 'Zelzlew', 'Afendaw', 'Agayew', 'Zerirew', 'Unknown', 'Silence']
    NUMBER_OF_SAMPLE_ANALYSED = 24000

    RECORD_SECONDS = 1.5
    SAMPLING_RATE = 16000
    frames = []
    WIDTH = 925
    HEIGHT = 555
    MODEL_PATH = "models/model.onnx"


    ks = Keyword_Spotter(WIDTH , HEIGHT,MODEL_PATH)

    while True:
        dt = clock.tick(60) / 1000.0  # time elapsed since last frame in seconds
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        ks.window.blit(ks.background, (0, 0))

        # record frames of audio data
        buff = stream.read(CHUNK)
        data = np.frombuffer(buff, dtype=np.int16)
        # frames.append(data)
        frames.append(data)
        arr1 = frames[0]
        arr2 = data
        con = np.hstack((arr1, arr2))
        frames.clear()
        frames.append(con)

        # stop recording after RECORD_SECONDS seconds
        elapsed_time = pygame.time.get_ticks() / 1000.0 - start_time
        # print(elapsed_time)
        if elapsed_time >= RECORD_SECONDS:
            #print(f"Recording done : {elapsed_time}")
            start_time = pygame.time.get_ticks() / 1000.0  # current time in seconds
            elapsed_time = 0
            # concatenate the arrays horizontally
            audio_data = np.hstack(frames)
            print(colored(f'[*] Length of Collected Sample : {len(audio_data)}', 'green'))

            #print("Length of Collected Sample : ", len(audio_data))

            #get_audio(audio_data)kk
            frames.clear()
            
            
            if len(audio_data) < NUMBER_OF_SAMPLE_ANALYSED:
                zeros_needed = NUMBER_OF_SAMPLE_ANALYSED - len(audio_data)
                audio_data = np.append(audio_data, np.zeros((zeros_needed)))
            
            
            if len(audio_data) >= NUMBER_OF_SAMPLE_ANALYSED:
                audio_data = audio_data[:NUMBER_OF_SAMPLE_ANALYSED].astype(np.float32)

                msLFB_spec = MSLFB(audio_data)
                mcff_spec = MFCC(audio_data)
                # Assume x_new_mfcc and x_new_mslfb are the new input data
                x_new_mfcc_reshaped_spec = np.reshape(mcff_spec, (1, mcff_spec.shape[0], mcff_spec.shape[1], 1)).astype(np.float32)
                x_new_mslfb_reshaped_spec = np.reshape(msLFB_spec, (1, msLFB_spec.shape[0], msLFB_spec.shape[1], 1)).astype(np.float32)


                # Make predictions on new data
                # Prepare the input data
                print("x_new_mfcc_reshaped_spec : ", x_new_mfcc_reshaped_spec.shape)
                print("x_new_mslfb_reshaped_spec : ", x_new_mslfb_reshaped_spec.shape)
                input_data = {'conv2d_18_input': x_new_mfcc_reshaped_spec, 'conv2d_21_input': x_new_mslfb_reshaped_spec}
                # Run the prediction
                try:
                    y_probs = self.model.run(None, input_data)
                    arr = y_probs[0]
                    selected_values = arr[arr > 0.5]

                    # Convert the output to class predictions
                    if len(selected_values) == 0:
                        result = "None"
                    else:
                        y_preds = selected_values.argmax()
                        result = f"{label_names[y_preds]}"
                except:
                    print("[!Notice!] Tensor Received incorrect Value".format(Fore.BLUE, Fore.RESET))
                    return None
            else:
                print(colored('[!] Keyspotter didnt capture enough frame , Free some RAM space', 'green'))

        ks.update(dt)  # You can update/draw here, I've just moved the code for neatness.
        ks.draw(data,result)
        pygame.display.flip()  # Refresh on-screen display
        clock.tick(60)


if __name__ == "__main__":
    main()
