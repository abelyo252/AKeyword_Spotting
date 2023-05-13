<p align="center"><img width="50%" src="https://github.com/abelyo252/ASP_Keyword_Spotting/assets/126100289/4a17064e-142d-4017-9c3c-b46235769d6b" /></p>

**AKeyword_Spotting is a audio signal processer and training used for keyspot from audio**.
This is a Amharic speech recognition model that makes its easy to run Criminal Speech recognition and AI functions. At the core it uses [Pygame](https://github.com/pygame) and [Onnxruntime](https://github.com/microsoft/onnxruntime) libraries. 


## Installation
You can  simply follow these instruction to install the latest version of ASP_Keyword_Detector.

`git clone https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting.git`<br>
`cd Terminal-ASP_Keyword_Spotting`<br>
`pip install -r requirements.txt`<br>

<hr>

### Average 0.89 sec for inference

<hr>

<p align="center">
  <img width="640" height="360" src="https://github.com/abelyo252/ASP_Keyword_Spotting/assets/126100289/97dd62ae-8bdd-459a-b0f1-0425d0a81811">
</p>


Keyword spotting is the process of detecting specific words or phrases in a continuous stream of speech. It is a crucial component of many speech recognition systems, including virtual assistants, voice-activated devices, and automated transcription tools. The goal of keyword spotting is to accurately identify and isolate specific keywords from a large corpus of speech data. It also include Graphical user interface


<p align="center">
  <img width="640" height="360" src="https://user-images.githubusercontent.com/126100289/234276539-81fe427d-eb08-4c58-b44d-0b82ad406b93.png">
</p>


##Dataset
The training data is prepare by Jimma University , 5th year Electrical and Computer Engineering Students, and contains 20 criminal voice mentioned above. All the data will be divided to 70% for training and 30% to validation in this work. Also, We have plan provided sample noises which were randomly selected and added to the train and validation sets to make the train and validation more real world like scenarios.


## Usage

You can use the project by running the `key_spotter.py` file and passing an Amharic text as input. The program will detect if any of the 20 criminal words are present in the text.

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
