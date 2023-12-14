import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Conv2D
from keras.layers import PReLU
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Dense
from keras.layers import Input
from keras.layers import add
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from keras.applications.vgg19 import VGG19
from keras.models import load_model
from numpy.random import randint

datadir = r"C:\Users\doant\Downloads\cars_train"

array = []
array_small =[]

def create_training_data():
        for img in tqdm(list(os.listdir(datadir))):
            try:
                img_array = cv2.imread(datadir+'/'+img ,cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (128, 128))
                array.append(new_array)
                array_small.append(cv2.resize(img_array, (32,32),
                               interpolation=cv2.INTER_CUBIC))
            except Exception as e:
                pass
create_training_data()

X =  []
Xs = []
for features in array:
    X.append(features)
for features in array_small:
    Xs.append(features)
plt.figure(figsize=(16, 8))
X = np.array(X).reshape(-1, 128, 128, 3)
Xs = np.array(Xs).reshape(-1, 32, 32, 3)
plt.subplot(231)
plt.imshow(X[0], cmap = 'gray')
plt.subplot(233)
plt.imshow(Xs[0], cmap = 'gray')

X_train,X_valid,y_train, y_valid = train_test_split(Xs, X, test_size = 0.33, random_state = 12)

def res_block(input_dim):
    model = Conv2D(64, (3,3), padding = 'same' )(input_dim)
    model = BatchNormalization()(model)
    model = PReLU(shared_axes = [1,2])(model)
    model = Conv2D(64, (3,3), padding = 'same' )(model)
    model = BatchNormalization()(model)
    return add([input_dim, model])
def upscale_block(input_dim):
    model = Conv2D(256,(3,3), strides=1, padding = 'same')(input_dim)
    model = UpSampling2D(size = (2,2))(model)
    model = PReLU(shared_axes=[1, 2])(model)
    return model
def generator(input, res_range = 1,upscale_range=1):
    model = Conv2D(64,(9,9), strides=1, padding = 'same')(input)
    model = PReLU(shared_axes = [1,2])(model)
    model1 = model
    for i in range(res_range):
        model = res_block(model)
    model = Conv2D(64, (3,3), padding = 'same' )(model)
    model = BatchNormalization()(model)
    model = add([model,model1])
    for i in range(upscale_range):
        model  =upscale_block(model)
    output = Conv2D(3, (9,9),  padding='same')(model)
    return Model(input, output)

def discrim_block(input_dim, fmaps = 64, strides = 1):
    model = Conv2D(fmaps, (3,3), padding = 'same', strides  = strides)(input_dim)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    return model
def discriminator(input):
    model = Conv2D(64,(3,3),padding='same')(input)
    model = LeakyReLU()(model)
    model = discrim_block(model, strides = 2)
    model = discrim_block(model, fmaps  = 128)
    model = discrim_block(model, fmaps = 128, strides = 2)
    model = discrim_block(model, fmaps=256)
    model = discrim_block(model, fmaps=256, strides=2)
    model = discrim_block(model, fmaps=512)
    model = discrim_block(model, fmaps=512, strides=2)
    model = Flatten()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    out = Dense(1, activation='sigmoid')(model)
    return Model(input, out)

def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


# Define combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)

    gen_features = vgg(gen_img)

    disc_model.trainable = False
    validity = disc_model(gen_img)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

hr_shape = (y_train.shape[1], y_train.shape[2], y_train.shape[3])
lr_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

generator = generator(lr_ip, res_range = 16, upscale_range=2)
generator.summary()

discriminator = discriminator(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
discriminator.summary()

vgg = build_vgg((128,128,3))
print(vgg.summary())
vgg.trainable = False

gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)

gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
gan_model.summary()

batch_size = 1
train_lr_batches = []
train_hr_batches = []
for it in range(int(y_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(y_train[start_idx:end_idx])
    train_lr_batches.append(X_train[start_idx:end_idx])

epochs = 1
for e in range(epochs):

    fake_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size, 1))

    g_losses = []
    d_losses = []

    for b in tqdm(range(len(train_hr_batches)), desc="Training..."):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]

        fake_imgs = generator.predict_on_batch(lr_imgs)

        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)

        discriminator.trainable = False
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
        image_features = vgg.predict(hr_imgs)
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
        d_losses.append(d_loss)
        g_losses.append(g_loss)

    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)

    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)

    print("epoch:", e + 1, "g_loss:", g_loss, "d_loss:", d_loss)

    if (e + 1) % 5 == 0:
        generator.save("gen_e_" + str(e + 1) + ".h5")

generator.save("generator_car"+ str(e+1) +".h5")

# generator = load_model('generator1.h5')
# path_img =r"C:\Users\doant\PycharmProjects\pythonProject1\ProjectDATN\Motor_Violations\27.jpg"

# src_image = cv2.imread(path_img)
# src_image_test = cv2.resize(src_image, (32,32))
# src_image_test = src_image_test.reshape((1,32,32,3))

[X1, X2] = [X_valid, y_valid]
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
gen_image = generator.predict(src_image)
# gen_image = generator.predict(src_image_test)
# gen_image = gen_image[0]

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Low Resolution Image')
# plt.imshow(src_image_test[0,:,:,:], cmap = 'gray')
plt.imshow(src_image[0,:,:,:], cmap = 'gray')
plt.subplot(232)
plt.title('Super Resolution Image')
plt.imshow(cv2.cvtColor(gen_image[0,:,:,:], cv2.COLOR_BGR2GRAY),cmap = 'gray')
# plt.imshow(cv2.cvtColor(gen_image, cv2.COLOR_BGR2GRAY),cmap = 'gray')
plt.subplot(233)
plt.title('Original High Resolution Image')
plt.imshow(tar_image[0,:,:,:], cmap = 'gray')
# plt.imshow(src_image, cmap = 'gray')

plt.show()