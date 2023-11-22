import time
from training import train_model
from testing import test_model
from os.path import join, exists
from os import makedirs
from torch.cuda import is_available
import sys


def main():
    use_gpu = sys.argv[1]

    # If you have to GPUs and the GPUs are not available
    if use_gpu == 'True' and not is_available():
        print('GPUs not available')
        exit()

    architecture = 1
    learning_rate = 0.001
    perc = 1
    prefix = "4"

    makedirs('results', exist_ok=True)
    folder_name = join('results', f"{architecture}_{learning_rate}_{prefix}")
    if exists(folder_name):
        print(f'This results folder already exists: {folder_name}'
              f'\nChanging the prefix by 1.')
	
	prefix = str(int(prefix) + 1)
	print(f'\nNew folder name: {folder_name}')
 
   makedirs(folder_name)

    print()
    t = time.time()
    model_file = train_model(color_dir=join('data', 'google-landmark', 'train'),
                             perc=perc,
                             folder_name=folder_name,
                             file_name=f"cnn_{prefix}",
                             architecture=architecture,
                             gray_dir=None,
                             epochs=30,
                             learning_rate=learning_rate,
                             model=None)

    t = time.time() - t
    print(f"Training ran for {time.strftime('%H:%M:%S', time.gmtime(t))}")
    print(f"\n{model_file} created.")

    print("\nTesting")
    test_model(model_file=model_file,
               color_dir=join('data', 'google-landmark', 'test'),
               perc=perc,
               folder_name=folder_name,
               file_name="cnn_testing",
               architecture=architecture,
               gray_dir=None,
               show_images=True)


if __name__ == "__main__":
    main()
