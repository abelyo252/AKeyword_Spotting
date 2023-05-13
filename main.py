import os

path = "train/audio/unknown/"

# get list of all files in the directory
files = os.listdir(path)

# loop through each file and rename it
for index, file in enumerate(files):
    # construct new file name
    new_file_name = "unknown_file_0" + str(index) + ".wav"

    # construct full paths for old and new file names
    old_path = os.path.join(path, file)
    new_path = os.path.join(path, new_file_name)

    # rename the file
    os.rename(old_path, new_path)
