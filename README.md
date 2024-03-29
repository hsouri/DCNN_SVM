# DCNN_SVM
Fashion MNIST classification using Deep Convolutional Neural Network and SVM with different kernels.

This is a Python implementation of a Convolutional Neural Network and a SVM classifer plus PCA and LDA dimension reduction on Fashion MNIST dataset.


Note1: Please download and copy fashion train and test set into the fashion folder of the data folder. Link: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz

Note2: This implementation is using Multi-GPUs for faster training. If you don't have gpu the code will automatically use CPU instead.

To see the loss and acuuracy of the network trainin and validation please take a look at `out.txt` file.


# Network Architecture
![Repo List](screenshots/Slide2.JPG)

The output of the network is 10 scores for each class.



# Getting Started
Clone this repository with the following command:

```shell
git clone https://github.com/hsouri/DCNN_SVM
```

# Usage

The SVM model trainng and testing can be done by running the main function in the Driver.py. Class SVM in the Classifiers package has the implementations of the SVM classifer.


# Train a SVM:

```shell
python Driver.py

```

How to read Fashion MNIST: 

```shell
traindata, trainlabels = mnistreader.loadmnist(’data/fashion’, kind=’train’)
testdata, testlabels = mnistreader.loadmnist(’data/fashion’, kind=’t10k’)
```

# DCNN Model training and Testing

- Training:

```shell
python DCNN.py
```

You can change attributes such as batch size, learning rate, number of epochs, number of workers, resume
and continue training from a checkpoint. List of selectable attributes:

'--name', '--out_file', '--workers', '--batch-size', '--resume', '--data', '--print_freq', '--epochs', '--start_epoch', '--save_freq'


# Continue Training

- Resume training from a saved model:

You are able to resume training your model from a saved checkpoint by running the following:

```shell
python DCNN.py --resume {directory path to your saved model}
```
To use my pretrained model, run the following:

```shell
python DCNN.py --resume 'saved_models/50_epoch_Fashion_MNIST_checkpoint.tar'
```


# Training and Test Loss
Losses for trainig the model will be saved in the `out.txt` file.

# Test Scores

Test scores are as follows:

![validation scores](screenshots/scores.PNG)
