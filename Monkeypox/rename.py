import os

# specify the folder containing the images
folder_path = './Monkeypox/6124_testing_dataset/Others/'

# get a list of all files in the folder
files = os.listdir(folder_path)

# iterate through the files and rename them
for i, file_name in enumerate(files):
    # split the file name and extension
    name, ext = os.path.splitext(file_name)
    # construct the new file name
    new_name = 'testing_ds_others_img_{}.{}'.format(i + 1, ext)
    # construct the full path of the old and new files
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)
    # rename the file
    os.rename(old_path, new_path)
    print('Renamed {} to {}'.format(file_name, new_name))