<p align="center"><img width="50%" src="https://github.com/abelyo252/ASP_Keyword_Spotting/assets/126100289/4a17064e-142d-4017-9c3c-b46235769d6b" /></p>

**AKeyword_Spotting is a audio signal processer and training used for keyspot from audio**.
This is a Amharic speech recognition model that makes its easy to run Criminal Speech recognition and AI functions. At the core it uses [Pygame](https://github.com/pygame) and [Onnxruntime](https://github.com/microsoft/onnxruntime) libraries. 


## Installation
You can  simply follow these instruction to install the latest version of ASP_Keyword_Detector. you must install python version between 3.6.x to 3.9.x

`git clone https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting.git`<br>
`cd Terminal-ASP_Keyword_Spotting`<br>
`pip install -r requirements.txt`<br>

<hr>

<p>Keyword spotting is the process of detecting specific words or phrases in a continuous stream of speech. It is a crucial component of many speech recognition systems, including virtual assistants, voice-activated devices, and automated transcription tools. The goal of keyword spotting is to accurately identify and isolate specific keywords from a large corpus of speech data. It also include Graphical user interface.</p>


<p align="center">
  <img width="640" height="360" src="https://github.com/abelyo252/ASP_Keyword_Spotting/assets/126100289/97dd62ae-8bdd-459a-b0f1-0425d0a81811">
</p>

<pre> Average 0.89 sec for inference</pre>



<p align="center">
  <img width="640" height="360" src="https://user-images.githubusercontent.com/126100289/234276539-81fe427d-eb08-4c58-b44d-0b82ad406b93.png">
</p>


## Dataset
The training data is prepare by Jimma University , 5th year Electrical and Computer Engineering Students, and contains 20 criminal voice mentioned above. All the data will be divided to 70% for training and 30% to validation in this work. Also, We have plan provided sample noises which were randomly selected and added to the train and validation sets to make the train and validation more real world like scenarios.


## Usage

You can use the project by running the `key_spotter.py` file and passing an Amharic text as input. The program will detect if any of the 20 criminal words are present in the text.

## How the Model Work

First Looking at the digital signal utterances here is how a few randomly selected ones look like in time domain with the word mentioned above each plot with their log spec:

<p align="center">
  <img width="640" height="360" src="https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting/assets/126100289/2954cd48-e2b0-42e3-a24a-8bc83076623c">
</p>

Of course, this is prior to the addition of any noise to the samples. Additionally, because the utterances rarely last a 3 full second, the length of them may be reduced depending on the strength of the signal. 

**Why MS-LFB**
Using deep learning to recognize speech automatically. Mel-Scaled Log Filter-Bank features (MS-LFB), which are both generated from the most often used raw features for speech recognition, are Mel Frequency Cepstral Coefficients (MFCC) and Perceptual Linear Predictive (PLP). According to experiments by this book, MS-LFB can perform better than MFCC when put to the test on a full multiple word utterance, with a relative Word Error Rate (WER) improvement of 4.4%. Consequently, MS-LFB is thought to be a strong contender to be used as a feature in this work.

** MS-LFB Architecture **
<p align="center">
  <img  src="https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting/assets/126100289/6a6f88ae-2cde-4c14-a4d6-523f9471fc2e">
</p>

**MS-LFB CNN Model**
A CNN model has been constructed based on the MS-LFB coefficients and using the 2D output of MS-LFB over time similar to a grayscale image. The image is fed into multiple layers of 2D convolution combined with pooling and dropout layers. Finally the layers have been flattened and with multiple dense layers predicted the output. The full model code is included in the ensemble model so it is not repeated here. But here is an image from reference [8] which describes this process:

<p align="center">
  <img width="640" height="360" src="https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting/assets/126100289/0709d3c8-6904-43cb-8d6b-872610a8045b">
</p>

**Over-ALL Architechure**
<p align="center">
  <img src="https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting/assets/126100289/f82c6c80-501e-4cbf-99ae-b432c3abbf6c">
</p>

## Run Code
`$ python key_spotter.py --al 48000 --model 'models/asp_ensemble_model.onnx' -- a audio.wav --conf 0.8`<br>

The output will be a list of detected criminal words, if any. with detected time stamp

## Voice Detection

To detect the 20 Amharic criminal words in voice samples, you can use the following steps:

1. Record a voice sample of the person speaking.
2. Convert the voice sample to a 16kHz mono WAV file.
3. Use the `key_spotter.py` program to detect if any of the 20 criminal words are present in the given audio file.


--- Special thanks to Adane T. for introducing this idea to us and help us in our difficulty in this project


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/abelyo252/ASP_Keyword_Spotting/issues).

For general discussion or questions, please use [GitHub Discussions](https://github.com/abelyo252/ASP_Keyword_Spotting/discussions).

## Contact

For more information contact [benyohanan212@gmail.com](mailto:benyohanan212@gmail.com) with any additional questions or comments.

<!Notice!>This project is still in demo mode, thus it does not guarantee a perfect result for the provided audio sample. With tremendous admiration, we are ready to accept anybody who can make a difference.
