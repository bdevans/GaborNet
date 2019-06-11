import os
import sys
import argparse
import json

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras
from keras import backend as K
from keras import activations, initializers, regularizers, constraints, metrics
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, Layer,
                          BatchNormalization, LocallyConnected2D,
                          ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose,
                          GaussianNoise, UpSampling2D, Input)
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils, multi_gpu_model
from keras.legacy import interfaces

from keras.layers import Lambda
import tensorflow as tf
import cv2
from keras.applications.vgg16 import VGG16


def load_images(path):

    image_set = {}
    for root, dirs, files in os.walk(path):
        if root == path:
            categories = sorted(dirs)
            image_set = {cat: [] for cat in categories}
        else:
            image_set[os.path.basename(root)] = sorted(files)

    n_cat_images = {cat: len(files) for (cat, files) in image_set.items()}
    n_images = sum(n_cat_images.values())
    image_dims = plt.imread(os.path.join(path, categories[0],
                            image_set[categories[0]][0])).shape

    print(image_dims)
    # X = np.zeros((n_images, *image_dims), dtype='float32')
    X = np.zeros((n_images, image_dims[0], image_dims[1], 1), dtype='float32')
    y = np.zeros((n_images, len(categories)), dtype=int)
    # y = np.zeros(n_images, dtype=int)

    tally = 0
    for c, (cat, files) in enumerate(tqdm(image_set.items(), desc=path)):
        for i, image in enumerate(files):
            cimg = plt.imread(os.path.join(path, cat, image))
            X[i+tally] = np.expand_dims(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY), axis=-1)
        y[tally:tally+len(files), c] = True
        tally += len(files)

    shuffle = np.random.permutation(y.shape[0])

    return image_set, X[shuffle], y[shuffle]


def get_dog_tensor(ksize, sigmas, r_sigmas):  # -> kx1tensor:
    # run a 5x5 gaussian blur then a 3x3 gaussian blr
    # blur5 = cv2.GaussianBlur(img,(5,5),0)
    # blur3 = cv2.GaussianBlur(img,(3,3),0)

    # s1 = sp.filter.gaussian_filter(img, k*sigma)
    # s2 = sp.filter.gaussian_filter(img, sigma)
    # dog = s1 - s2
    n_kernels = len(sigmas) * len(r_sigmas)
    kernels = []
    for sigma in sigmas:
        g_c = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_32F)
        for r_sigma in r_sigmas:
            g_s = cv2.getGaussianKernel(ksize, r_sigma*sigma, ktype=cv2.CV_32F)
            dog = g_c - g_s
            # dog /= np.sum(dog)  # Normalise
            dog = K.expand_dims(dog, -1)
            dog_on, dog_off = dog, dog * -1
            kernels.extend([dog_on, dog_off])
    assert len(kernels) == n_kernels
    print(f"Created {n_kernels} kernels.")
    return K.stack(kernels)


def get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis):

    n_kernels = len(sigmas) * len(thetas) * len(lambdas) * len(gammas) * len(psis)
    gabors = []
    for sigma in sigmas:
        for theta in thetas:
            for lambd in lambdas:
                for gamma in gammas:
                    for psi in psis:
                        params = {'ksize': ksize, 'sigma': sigma, 'theta': theta, 'lambd': lambd, 'gamma': gamma, 'psi': psi}
                        gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)
                        gf = K.expand_dims(gf, -1)
                        gabors.append(gf)
    assert len(gabors) == n_kernels
    print(f"Created {n_kernels} kernels.")
    return K.stack(gabors, axis=-1)


def convolve_tensor(x, kernel_tensor=None):
    '''
    conv2d
    input tensor of shape [batch, in_height, in_width, in_channels]
    kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    '''
    # x = tf.image.rgb_to_grayscale(x)
    return K.conv2d(x, kernel_tensor, padding='same')


# def lambda_output_shape(input_shape):
#     return input_shape


# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--data_set', type=str, default='cifar10',
                    help='Data set to use')
parser.add_argument('--stimulus_set', type=str, default='static',
                    help='Stimulus set to use')
parser.add_argument('--noise_type', type=str, default=None,
                    help='Noise mask to use')
parser.add_argument('--trial_label', default='Trial1',
                    help='For labeling different runs of the same model')
parser.add_argument('--filter_type', type=str, default='gabor',
                    help='Convolutional filter type')
parser.add_argument('--filter_size', type=int, default=31,
                    help='Convolutional filter size')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train model')
parser.add_argument('--data_augmentation', type=int, default=1,
                    help='Flag to use data augmentation in training')
parser.add_argument('--fresh_data', type=int, default=0,
                    help='Flag to (re)read images from files')
parser.add_argument('--model_name', type=str, default=None,
                    help='File name root to save outputs with')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='Pretrained model')
parser.add_argument('--n_gpus', type=int, default=1,
                    help='Number of GPUs to train across')
parser.add_argument('--save_loss', type=int, default=0,
                    help='Flag to save loss')
parser.add_argument('--lambd', type=float, default=None,
                    help='Gabor sinusoid wavelength')
parser.add_argument('--sigma', type=float, default=None,
                    help='Gabor/DoG Gaussian envelope standard deviation')
parser.add_argument('--r_sigma', type=float, default=None,
                    help='DoG ratio of standard deviations: sigma_s/sigma_c')

args = parser.parse_args()

data_set = args.data_set
stimulus_set = args.stimulus_set
noise_type = args.noise_type
trial_label = args.trial_label
filter_type = args.filter_type.lower()
filter_size = args.filter_size
epochs = args.epochs
data_augmentation = args.data_augmentation
fresh_data = args.fresh_data
model_name = args.model_name
pretrained_model = args.pretrained_model
n_gpus = args.n_gpus
save_loss = args.save_loss
lambd = args.lambd
sigma = args.sigma

weights = None  # 'imagenet'
input_shape = (32, 32, 1)  # (224, 224, 3)

project_root = os.path.realpath(os.pardir)
# save_dir = os.path.join(os.getcwd(), 'results')  # TODO: /workspace/results
save_dir = os.path.join(project_root, 'results', filter_type, data_set, stimulus_set)
# data_set = 'pixel'
data_root = '/work/data/pixel/small'  # TODO: Pass in
# stimulus_set = 'static'  # 'jitter'  # 'static'  # 'set_32_32'
if noise_type is None:
    noise_types = ['Original', 'Salt-and-pepper', 'Additive', 'Single-pixel']
else:  # Run with a single noise mask
    noise_types = [noise_type]
test_conditions = ['Same', 'Diff', 'NoPix']

pretrained_model = False
data_augmentation = False


if filter_type.lower() == 'dog':
    # Difference of Gaussian parameters
    ksize = filter_size  # 31
    if sigma:
        sigmas = [sigma]
    else:
        sigmas = [1, 2, 3, 4]
    if r_sigma:
        r_sigmas = [r_sigma]
    else:
        r_sigmas = [1.1, 1.2, 1.5, 2, 2.5, 3]

    # Generate DoG filters
    tensor = get_dog_tensor(ksize, sigmas, r_sigmas)

elif filter_type.lower() == 'gabor':
    # Gabor filter parameters
    ksize = (filter_size, filter_size)  # (31, 31)
    if sigma:
        sigmas = [sigma]
        n_orients = 8
    else:
        sigmas = [2, 4]
        n_orients = 4
    thetas = np.linspace(0, np.pi, n_orients, endpoint=False)  # [0, np.pi/4, np.pi/2, np.pi*3/4]
    if lambd:
        lambdas = [lambd]
    else:
        lambdas = [8, 16, 32, 64]
    n_phases = 4  # 1, 2, 4
    psis = np.linspace(0, 2*np.pi, n_phases, endpoint=False)  # [0, np.pi/2, np.pi, 3*np.pi/2]  # [np.pi/2]
    n_ratios = 2  # 1, 2, 4
    gammas = np.linspace(1, 0, n_ratios, endpoint=False)

    # Generate Gabor filters
    tensor = get_gabor_tensor(ksize, sigmas, thetas, lambdas, gammas, psis)

else:
    sys.exit(f"Unknown filter type requested: {filter_type}")

# fresh_data = True
batch_size = 64
num_classes = 10

for noise_type in noise_types:

    # model_name = f"{data_set}_{stimulus_set}_{noise_type}_{trial_label}"
    model_name = f"{noise_type}_{trial_label}"
    print("Running model:", model_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model_path = os.path.join(save_dir, model_name)

    # if data_set == 'cifar10':
    #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #     x_train = np.mean(x_train, 3, keepdims=True)  # Average over RGB channels
    #     x_test = np.mean(x_test, 3, keepdims=True)  # Average over RGB channels
    #     x_train = x_train.astype('float32')
    #     x_test = x_test.astype('float32')
    #     x_train /= 255
    #     x_test /= 255

    #     # Convert class vectors to binary class matrices.
    #     y_train = keras.utils.to_categorical(y_train, num_classes)
    #     y_test = keras.utils.to_categorical(y_test, num_classes)

    # elif data_set == 'pixel':

    #   (noise_type, trial) = trial_label.split("_")

    if noise_type == 'Original':
        data_path = os.path.join(data_root, stimulus_set, 'orig')
    elif noise_type == 'Salt-and-pepper':
        data_path = os.path.join(data_root, stimulus_set, 'salt_n_pepper')
    elif noise_type == 'Additive':
        data_path = os.path.join(data_root, stimulus_set, 'uniform')
    elif noise_type == 'Single-pixel':
        data_path = os.path.join(data_root, stimulus_set, 'single_pixel')
    else:
        sys.exit(f"Unknown noise type requested: {noise_type}")

    train_path = os.path.join(data_path, 'train')
    # test_path = os.path.join(data_path, f"test_{noise_cond.lower()}")

    if os.path.isfile(os.path.join(train_path, 'x_train.npy')) and not fresh_data:
        print(f'Loading {data_set} data arrays.')
        x_train = np.load(os.path.join(train_path, 'x_train.npy'))
        y_train = np.load(os.path.join(train_path, 'y_train.npy'))
        # num_classes = len(os.listdir(train_path)) - 1
        cat_dirs = [os.path.join(train_path, o) for o in os.listdir(train_path)
                    if os.path.isdir(os.path.join(train_path, o))]
        assert num_classes == len(cat_dirs)
    else:
        print(f'Loading {data_set} image files.')
        train_images, x_train, y_train = load_images(train_path)
        print(train_images.keys())
        assert num_classes == len(train_images)
        np.save(os.path.join(train_path, 'x_train.npy'), x_train)
        np.save(os.path.join(train_path, 'y_train.npy'), y_train)

    # x_train = np.mean(x_train, 3, keepdims=True)  # Average over RGB channels

    test_sets = []
    for test_cond in test_conditions:
        test_path = os.path.join(data_path, f"test_{test_cond.lower()}")
        if os.path.isfile(os.path.join(test_path, 'x_test.npy')) and not fresh_data:
            x_test = np.load(os.path.join(test_path, 'x_test.npy'))
            y_test = np.load(os.path.join(test_path, 'y_test.npy'))
        else:
            test_images, x_test, y_test = load_images(test_path)
            print(test_images.keys())
            assert num_classes == len(test_images)
            np.save(os.path.join(test_path, 'x_test.npy'), x_test)
            np.save(os.path.join(test_path, 'y_test.npy'), y_test)
        # test_sets.append((np.mean(x_test, 3, keepdims=True), y_test))
        test_sets.append((x_test, y_test))
    test_cond = "NoPix"  # Use this for examining learning curves
    x_test, y_test = test_sets[test_conditions.index("NoPix")]  # Unpack default test set
    # else:
    #     sys.exit(f"Unknown data set requested: {data_set}")

    # Summarise stimuli
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(y_train.shape[1], 'training categories')
    print(y_test.shape[1], 'testing categories')

    # Import VGG16
    model = VGG16(include_top=True, weights=weights, input_tensor=None,
                  input_shape=input_shape, pooling=None, classes=num_classes)

    # Modify standard VGG16 with hardcoded Gabor convolutional layer
    layers = [l for l in model.layers]
    # x = layers[0].output
    inp = Input(shape=x_train[0].shape)
    x = Lambda(convolve_tensor, arguments={'kernel_tensor': tensor})(inp)
    for l in range(2, len(layers)):
        x = layers[l](x)

    model = Model(inputs=inp, outputs=x)
    # model.summary()

    if pretrained_model:
        # Load weights from saved model
        pretrained_model_path = os.path.join(save_dir, pretrained_model)
        model.load_weights(pretrained_model_path, by_name=True)

        # Freeze weights in convolutional layers during training
        for layer in model.layers:
            if isinstance(layer, keras.layers.convolutional.Conv2D):
                print(f"Freezing layer: {layer.name}")
                layer.trainable = False

    if n_gpus > 1:
        model = multi_gpu_model(model, gpus=n_gpus)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Compile the model last before training for all changes to take effect
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    if not data_augmentation:
        print('Not using data augmentation.')
        hist = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(x_test, y_test),
                         shuffle=True)

    else:
        print('Using data augmentation.')
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        datagen.fit(x_train)

        hist = model.fit_generator(datagen.flow(x_train, y_train,
                                                batch_size=batch_size),
                                   steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                                   epochs=epochs,
                                   validation_data=(x_test, y_test),
                                   workers=4)

    print('History', hist.history)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model_name = 'SAVED'+'_'+model_name
    model_path = os.path.join(save_dir, model_name)
    # model.save(model_path)
    np.save(os.path.join(save_dir, f'{model_name}_VALACC.npy'), hist.history['val_acc'])
    np.save(os.path.join(save_dir, f'{model_name}_ACC.npy'), hist.history['acc'])
    if save_loss:
        np.save(os.path.join(save_dir, f'{model_name}_VALLOSS.npy'), hist.history['val_loss'])
        np.save(os.path.join(save_dir, f'{model_name}_LOSS.npy'), hist.history['loss'])

    cond_acc = {}
    cond_loss = {}
    for test_cond, (x_test, y_test) in zip(test_conditions, test_sets):
        loss, val_acc = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
        cond_acc[test_cond] = val_acc
        cond_loss[test_cond] = loss
    print("Saving metrics: ", model.metrics_names)
    with open(os.path.join(save_dir, f'{model_name}_CONDVALACC.json'), "w") as jf:
        json.dump(cond_acc, jf)
    if save_loss:
        with open(os.path.join(save_dir, f'{model_name}_CONDVALLOSS.json'), "w") as jf:
            json.dump(cond_loss, jf)

    print(f'Saved trained model at {model_path}')
