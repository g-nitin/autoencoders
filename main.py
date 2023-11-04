import time
from training import train_model
from testing import test_model
from os.path import join, exists
from os import makedirs

architecture = 7
learning_rate = 0.001
perc = 0.01/10
prefix = "3"

makedirs('results', exist_ok=True)
folder_name = join('results', f"{architecture}_{learning_rate}_{prefix}")
if exists(folder_name):
    print(f'This results folder already exists: {folder_name}'
          f'\nTry changing the name to avoid overriding')
    exit()
makedirs(folder_name)

print()
t = time.time()

model_file = train_model(color_dir=join('..', 'autoencoders-pg', 'data_cleaned', 'data_600_800.1', 'train'),
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

perc = 0.01
print("\nTesting")
test_model(model_file=model_file,
           color_dir=join('..', 'autoencoders-pg', 'data_cleaned', 'data_600_800.1', 'test'),
           perc=perc,
           folder_name=folder_name,
           file_name="cnn_testing",
           architecture=architecture,
           gray_dir=None,
           show_images=True)
