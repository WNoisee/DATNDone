from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Reshape, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tqdm import tqdm
import os
import cv2

path = r"C:\Users\doant\Dropbox\fruits-360_dataset\fruits-360\Training\Apple Braeburn"
path_img = r"C:\Users\doant\PycharmProjects\pythonProject7\ProjectDATN\Car_Violations"

data_gray = []
for i in tqdm(os.listdir(path_img), desc="preprocessing"):
    file = os.path.join(path_img, i)
    img = cv2.imread(file)
    img = cv2.resize(img, (150,150))
    data_gray.append(img)

np.save("data_gray.npy", data_gray)

# (X_train, _), (X_test, _) = mnist.load_data()
# X = np.vstack((X_train, X_test))
# X = X.astype("float32")
# X_train = (X-127) / 127

X_train = np.load("data_gray.npy")
X_train = X_train.astype("float32")
X_train = X_train / 255
#
def discriminator_dense():
    inp = Input(shape=(150,150,3))

    x = Conv2D(filters=16, kernel_size=(3,3))(inp)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    op = Dense(1, activation="sigmoid")(x)

    model = Model(inp, op)
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    return model

def generator(n):
    inp = Input(shape=(n))

    x = Dense(256, activation=LeakyReLU(alpha=0.2))(inp)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(67500, activation="tanh")(x)

    op = Reshape((150,150,3))(x)

    model = Model(inp, op)
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    return model

def gan(discrim, gen):
    discrim.trainable = False

    model = Sequential()
    model.add(gen)
    model.add(discrim)

    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    return model

discrim = discriminator_dense()

gener = generator(100)

gan_model = gan(discrim, gener)

epochs = 200
batch_size = 256
half_batch_size = batch_size // 2
n = 100
losses = []

for i in tqdm(range(epochs), desc="Processing: "):
    print("epoch: ", i)
    for j in range(len(X_train)//batch_size):
        idx_real = np.random.randint(0, len(X_train), half_batch_size)
        xreal, yreal = X_train[idx_real].reshape(half_batch_size, 150, 150, 3), np.ones((half_batch_size, 1))

        noise = np.random.randn(half_batch_size, n)
        xfake, yfake = gener.predict(noise), np.zeros((half_batch_size, 1))

        xfinal, yfinal = np.vstack((xreal, xfake)), np.vstack((yreal, yfake))

        indices = np.arange(len(yfinal))
        np.random.shuffle(indices)
        xfinal, yfinal = xfinal[indices], yfinal[indices]
        dloss = discrim.train_on_batch(xfinal, yfinal)

        gloss = gan_model.train_on_batch(np.random.randn(batch_size, n), np.ones(batch_size).reshape(batch_size, 1))

        losses.append([dloss, gloss])
        print(f"DLoss: {dloss}, GLoss: {gloss}")

fig, axis = plt.subplots(5,5,figsize=(12,12))
for ii in range(5):
    for jj in range(5):
        axis[ii, jj].imshow(gener.predict(np.random.randn(1*n).reshape(1, n)).reshape(150,150,3), cmap="gray")

plt.show()
plt.close()


