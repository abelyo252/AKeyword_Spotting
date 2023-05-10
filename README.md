# Terminal-ASP_Keyword_Spotting

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/f/fe/Current_Logo_of_Jimma_University.png/220px-Current_Logo_of_Jimma_University.png" />
</p>

## JIMMA UNIVERSITY
 Electrical and Computer Engineering 5th Year Assignment
 
Terminal Keyword spotting is a speech recognition model that can detect 20 criminal Amharic words. This model can be useful in law enforcement and security applications where it is necessary to detect certain keywords in spoken language.

# Project Description
This is a project to detect the 20 Amharic following 2-3 second criminal voice sampled at 16kHz : Eserat , Agtew, Dferat, Tlefat , Reshinachew , Tsetargew, Forjid , Shibr , Gejera , Ets , Gubo , Zrefew , Refrfew , Dfaw , Selilew , Musina , Zelzlew , Afendaw , Agayew , Zerirew.


#Dataset
The training data is prepare by Jimma University , 5th year Electrical and Computer Engineering Students, and contains 20 criminal voice mentioned above. All the data will be divided to 70% for training and 30% to validation in this work. Also, We have plan provided sample noises which were randomly selected and added to the train and validation sets to make the train and validation more real world like scenarios.


## Installation

To use the project, you need to clone the repository and install the required dependencies.<br>
`$ git clone https://github.com/abelyo252/Terminal-ASP_Keyword_Spotting.git`<br>
`$ cd Terminal-ASP_Keyword_Spotting`<br>
`$ pip install -r requirements.txt`<br>

## Usage

You can use the project by running the `key_spotter.py` file and passing an Amharic text as input. The program will detect if any of the 20 criminal words are present in the text.

## Run Code
`$ python key_spotter.py --al 48000 --model 'models/asp_ensemble_model.onnx`<br>

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
