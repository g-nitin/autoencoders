import subprocess
import os
import shutil
from torchvision.io import read_image
import sys

def main():

    # num_files = get_num_files()
    # num_files = ('010', '01')
    num_files = (sys.argv[1], sys.argv[2])
    main_dir = os.getcwd()
    print()

    if not os.path.exists('google-landmark'):
        os.makedirs('google-landmark')
        subprocess.run(['git', 'clone', 'https://github.com/cvdfoundation/google-landmark.git'])
        print('google-landmark directory created\n')

    train_path = os.path.join('google-landmark', 'train')
    os.makedirs(train_path, exist_ok=True)
    test_path = os.path.join('google-landmark', 'test')
    os.makedirs(test_path, exist_ok=True)
    print('Train and Test directories created under google-landmark\n')

    script_path = os.path.join('..', 'download-dataset.sh')  # Path for the download script

    os.chdir(train_path)
    subprocess.run(['bash', script_path, 'train', num_files[0]])
    print('Downloaded training images\n')
    os.chdir(main_dir)  # Go back

    os.chdir(test_path)
    subprocess.run(['bash', script_path, 'test', num_files[1]])
    print('Downloaded test images\n')
    os.chdir(main_dir)

    # Since the train and test folders contain images within many sub-folders,
    # Get all images and move them to a single level folder...
    train_path_new = os.path.join('google-landmark', 'train_new')
    os.makedirs(train_path_new, exist_ok=True)
    test_path_new = os.path.join('google-landmark', 'test_new')
    os.makedirs(test_path_new, exist_ok=True)

    print(f"Relocated {relocate_jpg_files(train_path, train_path_new)} training files.")
    print(f"Relocated {relocate_jpg_files(test_path, test_path_new)} testing files.\n")

    # Remove the original folders
    subprocess.run(['rm', '-rf', train_path])
    print(f'Removed {train_path}')
    subprocess.run(['rm', '-rf', test_path])
    print(f'Removed {test_path}\n')

    # Keep only those images which have a 600*800 size...
    # Get a list of paths for each images
    train = [os.path.join(train_path_new, file) for file in os.listdir(train_path_new)]
    test = [os.path.join(test_path_new, file) for file in os.listdir(test_path_new)]

    # Create dictionaries to store the image paths and their corresponding shapes
    train_dict = get_shape_paths(train, desired_shape=(3, 600, 800))
    test_dict = get_shape_paths(test, desired_shape=(3, 600, 800))

    # Move these files to the original folders
    os.makedirs(train_path)
    os.makedirs(test_path)
    print(f"Relocated {relocate_from_dict(train_dict, train_path)} training images of size 600 * 800")
    print(f"Relocated {relocate_from_dict(test_dict, test_path)} testing images of size 600 * 800\n")

    # Remove the new folders
    subprocess.run(['rm', '-rf', train_path_new])
    print(f'Removed {train_path_new}')
    subprocess.run(['rm', '-rf', test_path_new])
    print(f'Removed {test_path_new}')


def get_num_files():
    """
    Return the number of TAR files to download for training and testing by the user's input
    :return: tuple for the number of training files and the number of testing files.
    """
    train_num = input('Enter the number training TAR files to fetch (000-499): ')
    test_num = input('Enter the number testing TAR files to fetch (00-19): ')

    return str(train_num), str(test_num)


def relocate_jpg_files(src, dst, move=True):
    """
    Move or Cpy images from the `src` directory to the `dst` directory.
    This is done recursively, meaning that the function will move (or copy) all files in all subdirectories of `src`.
    :param src: The source directory
    :param dst: The final directory
    :param move: Whether to move files (True) or copy them (False)
    :return: The number of files moved
    """
    num = 0
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.lower().endswith('.jpg'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst, file)

                if move:
                    shutil.move(src_file, dst_file)
                else:
                    shutil.copy(src_file, dst_file)
                num += 1

    return num


def get_shape_paths(paths, desired_shape):
    """
    Given a list of image paths, return a dictionary of the path (key) and its shape (value).
    Only return those images which have the `desired_shape`.
    Using shape() from torchvision.io.read_image
    :param paths: A list of image paths
    :param desired_shape: A tuple for the desired shape
    :return: dict of paths to their shapes
    """
    paths_dict = {}

    for path in paths:
        # Read the image and get its shape
        shape = read_image(path).shape

        # Check if the shape matches the desired shape
        if shape == desired_shape:
            # Store the shape in the dictionary with the image path as the key
            paths_dict[path] = shape

    return paths_dict


def relocate_from_dict(img_dict, new_path, move=True):
    """
    Relocate files given in the keys of `img_dict` to the `new_path`.
    :param img_dict: dict for the images (as keys)
    :param new_path: str for the new path for those images
    :param move: bool for whether to move (True) or copy (False)
    :return: int indicating the number of files moved
    """
    num = 0
    for path in img_dict.keys():
        file = os.path.split(path)[1]

        if move:
            shutil.move(path, os.path.join(new_path, file))
        else:
            shutil.copy(path, os.path.join(new_path, file))

        num += 1
    return num


if __name__ == '__main__':
    main()
