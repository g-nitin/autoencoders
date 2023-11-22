# Autoencoders

The code (running `main.py`) will generate a `results` folder which has sub-folders for each new model test.
To avoid pushing large amounts of data (in the `results` sub-folder), the sub-folder has been ignored.

The `data` folder contains the `data_builder.py` file, which uses the [google-landmark](https://github.com/cvdfoundation/google-landmark) dataset to fetch images of size 600 by 800.
To get the data, just run the `data_build.py` file, which will create a sub-folder (`google-landmark`) in `data`.
That sub-folder contains the `train` and `test` images in those sub-folders.

If you are implementing another architecture, then it must be defined in `network.py` and subsequent changes must be made in `utilities.py`.