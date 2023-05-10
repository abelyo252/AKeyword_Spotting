# Terminal-ASP_Keyword_Spotting
Terminal Keyword spotting is a speech recognition model that can detect 20 criminal Amharic words. This model can be useful in law enforcement and security applications where it is necessary to detect certain keywords in spoken language.

## Installation

To use the project, you need to clone the repository and install the required dependencies.

sh
$ git clone https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting.git
$ cd Terminal-ASP_Keyword_Spotting
$ pip install -r requirements.txt


## Usage

You can use the project by running the `key_spotter.py` file and passing an Amharic text as input. The program will detect if any of the 20 criminal words are present in the text.

sh
$ python key_spotter.py --al 48000 --model 'models/asp_ensemble_model.onnx'


The output will be a list of detected criminal words, if any. with detected time stamp

## Voice Detection

To detect the 20 Amharic criminal words in voice samples, you can use the following steps:

1. Record a voice sample of the person speaking.
2. Convert the voice sample to a 16kHz mono WAV file.
3. Use the `key_spotter.py` program to detect if any of the 20 criminal words are present in the given audio file.

## Contributors
List of Contributors on preparing Dataset from Jimma University Electrical and Computer Engineering Student are

    Abel Yohannes
    Abel Demis
    Abubaker Zerihun
    Adem Seid
    Frehiwot Asres
    Hilina
    Eshcol
    Jerusalem
    Jerusalem
    Tsega
    Frezer
    Gadise
    Ba
    Nahom Diro
    Ruth Alemalew

--- Special thanks to Adane T. for introducing this idea to us and help us in our difficulty in this project

## Contributing

If you want to contribute to this project, please follow these guidelines:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Write clean and readable code.
3. Write tests for your code.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the Jimma University License - see the [LICENSE](LICENSE) file for details.

<!Notice!>This project is still in demo mode, thus it does not guarantee a perfect result for the provided audio sample. With tremendous admiration, we are ready to accept anybody who can make a difference.
