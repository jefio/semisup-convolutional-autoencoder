# Semi-supervised Convolutional Autoencoders

## Dependencies

Python 3.6, Numpy, Pandas, and Matplotlib. You will also need to install this [modified version of Keras](https://github.com/jefio/keras/tree/feature/class-mode-xxy).

## Setup

Prepare the dataset:
1. Download and unzip the [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). This creates a folder named `PetImages`
2. `cd /path/to/semisup-convolutional-autoencoder`
3. `. create_dir.sh /path/to/PetImages`

## Example

```
python run_cae.py
```
