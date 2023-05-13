"""
Keyword Spotting
By: Jimma University , Electrical and computer Enginnering Student
Dataset Collected : 2011 Batch Computer Stream Student
Website: https://github.com/abelyo252/AKeyword_Spotting

"""
# Import standard modules.
import argparse
import os
import time
import wave
from itertools import zip_longest

# Import non-standard modules.
import numpy as np
import librosa
import pyfiglet
from halo import Halo
from colorama import Fore, init
import onnxruntime
from utils import log_specgram , MFCC , MSLFB
import tqdm




class Keyword_Spotter:
    def __init__(self, model_path, audio_length, audio_path,dec_conf):
        self.model_path = model_path
        self.audio_length = audio_length
        self.audio_path = audio_path
        self.dec_conf = dec_conf

        self.RATE = 16000
        self.CHUNK_SIZE = self.audio_length
        self.HALF_CHUNK_SIZE = self.CHUNK_SIZE // 2

        self.chunks = []
        self.new_chunks = []
        self.result1 = []
        self.result2 = []
        self.classes = ['Eserat', 'Agtew', 'Dferat', 'Tlefat', 'Reshinachew', 'Tsetargew',
                   'Forjid', 'Shibr', 'Gejera', 'Ets', 'Gubo', 'Zrefew', 'Refrfew',
                   'Dfaw', 'Selilew', 'Musina', 'Zelzlew', 'Afendaw', 'Agayew', 'Zerirew', 'Unknown', 'Silence']

        # Call User Defined Functions
        self.display_init()
        self.audio, self.sr = self.get_audio()
        self.model = self.get_model()

    def display_init(self):
        try:
            print(pyfiglet.figlet_format('Keyword Spotter', justify='center'))
            print("""\t{}______
                                Version 1.0
                                Created By : Jimma University , Electrical and Computer Enginnering                           
                                                        {}\n""".format(Fore.BLUE, Fore.RESET))
            time.sleep(5)
            print("Welcome to Keyword Spotter!")
            self.syntax_helper()
            print()
        except Exception as e:
            return "[-] Couldn't show the start painting: {}".format(e)

    def get_audio(self):
        # read in the audio data and resample it
        audio, sr = librosa.load(self.audio_path, sr=self.RATE)
        return audio, sr

    def get_model(self):
        try:
            sess = onnxruntime.InferenceSession(self.model_path)
            return sess
        except Exception as e:
            return "[-] Couldn't get model or Model is Invalid: {}".format(e)

    def syntax_helper(self):
        print("Syntax: python key_spotter.py --al <audio_length> --model <model_path> --a <audio_path> --conf <detection_confidence>")

    def file_checker(self):

        if self.audio_length != 48000:
            print("\033[91mInvalid Audio Length Passed , Model prepared for analysed 48000 sample!\033[0m")
            self.syntax_helper()
            exit()

        if not 0 <= self.dec_conf <= 1:
            print("\033[91mInvalid Detection Confidence passed!\033[0m")
            self.syntax_helper()
            exit()

            
        if not os.path.isfile(self.model_path):
            print("\033[91mInvalid model path or file not found!\033[0m")
            self.syntax_helper()
            exit()

        if not os.path.isfile(self.audio_path) or not self.audio_path.endswith(".wav"):
            print("\033[91mInvalid audio path or file not found!\033[0m")
            self.syntax_helper()
            exit()

        
    def audio_chunker(self):

        # divide the audio into chunks of 48000 samples
        for i in range(0, len(self.audio), self.CHUNK_SIZE):
            self.chunks.append(self.audio[i:i + self.CHUNK_SIZE])

        for i in range(len(self.chunks) - 1):
            new_concat = np.concatenate(
                (self.chunks[i][self.HALF_CHUNK_SIZE:], self.chunks[i + 1][:self.HALF_CHUNK_SIZE]))
            self.new_chunks.append(new_concat)


    def predict_model(self, signal_resampled):


        audio_length = 48000
        if len(signal_resampled.shape) == 2:
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
        x_new_mfcc_reshaped_spec = np.reshape(mcff_spec, (1, mcff_spec.shape[0], mcff_spec.shape[1], 1)).astype(
            np.float32)
        x_new_mslfb_reshaped_spec = np.reshape(msLFB_spec, (1, msLFB_spec.shape[0], msLFB_spec.shape[1], 1)).astype(
            np.float32)

        # Make predictions on new data
        # Prepare the input dat

        # Make predictions on new data
        # Prepare the input data
        input_data = {'conv2d_18_input': x_new_mfcc_reshaped_spec, 'conv2d_21_input': x_new_mslfb_reshaped_spec}
        # Run the prediction
        try:
            y_probs = self.model.run(None, input_data)
            arr = y_probs[0]
            selected_values = arr[arr > self.dec_conf]

            # Convert the output to class predictions
            if len(selected_values) == 0:
                return None

            else:
                y_preds = selected_values.argmax()
                return self.classes[y_preds]
        except:
            print("[!Notice!] Tensor Received incorrect Value".format(Fore.BLUE, Fore.RESET))
            return None

    def process_asp(self):
        # Call model to process all chunk from the list
        time = 0
        for audio_data in tqdm.tqdm(self.chunks, desc="Processing audio chunks"):
            res = self.predict_model(audio_data)
            if res == None:
                continue
            self.result1.append(res)

        time = 2
        for audio_data in tqdm.tqdm(self.new_chunks, desc="Processing audio chunks"):
            res = self.predict_model(audio_data)
            if res == None:
                continue
            self.result2.append(res)

    def display_result(self,infer):
        # display the result on terminal
        print()
        print(
            """{}[*] We have detected {} Sequencial and {} Intermediatary Amharic criminal words in this audio file [*]{}\n""".format(
                Fore.GREEN, len(self.result1), len(self.result2), Fore.RESET))
        print("=============================================================================")
        print()
        time1 = 0
        time2 = 2
        for disp in zip_longest(self.result1,self.result2):
            
                minute1 = int(time1 / 60)
                second1 = time1 % 60
                minute2 = int(time2 / 60)
                second2 = time2 % 60
                print(f"onDetect Seq: {disp[0]} -> around {minute1:02d}:{second1:02d},  Intermediatary: {disp[1]} -> around {minute2:02d}:{second2:02d}")
                time1 = time1 + 3
                time2 = time2 + 3


        print(f"")
        print(
            """{}[*] Model Inference time for the given audio: {} second [*]{}\n""".format(
                Fore.GREEN, str( round(infer, 3)), Fore.RESET))
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--al", type=int, help="audio length")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--a", type=str, help="audio path")
    parser.add_argument("--conf", type=float, help="Detection Confidence")
    args = parser.parse_args()

    ks = Keyword_Spotter(args.model, args.al, args.a , args.conf)
    ks.file_checker()
    ks.audio_chunker()
    ptime = time.time()
    ks.process_asp()
    ctime = time.time()

    infer = ctime - ptime
    ks.display_result(infer)


if __name__ == '__main__':
    main()

# The whole code is available at https://github.com/abelyo252/ASP_Keyword_Spotting
# Issue if there is an error you will encounter
