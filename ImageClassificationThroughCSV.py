from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import backend as K, optimizers, callbacks

import numpy as np
import os.path


tensorboard=callbacks.TensorBoard(log_dir='C:\\Users\\Vivaswan Chandru\\Box Sync\\Mayanka-Research\\2018-CrisisCNN-WordEmbedding\\Code\\Keras_ImageClassification\\logs', histogram_freq=0, batch_size=10, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

img_width = 2
img_height = 4
NUM_CHANNELS = 3
if not os.path.isfile("data/im.npy"):
    pos = np.loadtxt('data/im1.csv', delimiter=',', dtype=np.float32)
    num_images = 77

    pos=pos.reshape(num_images,NUM_CHANNELS,img_height, img_width)
    np.save('data/im.npy', pos);
else:
    pos = np.load('data/im.npy')

if not os.path.isfile("data/cp.npy"):
    neg = np.loadtxt('data/cp1.csv', delimiter=',', dtype=np.float32)
    num_images=143
    neg=neg.reshape(num_images,NUM_CHANNELS, img_height, img_width)
    np.save('data/cp.npy', neg);
else:
    neg = np.load('data/cp.npy')


#if K.image_data_format() == 'channels_first':
input_shape = (NUM_CHANNELS, img_height, img_width)
# else:
#     input_shape = (img_height, img_width, 2)

model = Sequential()
model.add(Conv2D(128, (1, 1), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])


pos_labels = np.ones((pos.shape[0], 1), dtype=int);
neg_labels = np.zeros((neg.shape[0], 1), dtype=int);

print("positive samples: ", pos.shape[0])
print("negative samples: ", neg.shape[0])

model.fit(np.vstack((pos, neg)), np.vstack((pos_labels, neg_labels)), epochs=10, batch_size=10,callbacks=[tensorboard])

# true positive rate
tp = np.sum(model.predict_classes(pos))
tp_rate = float(tp)/pos.shape[0]

# false positive rate
fp = np.sum(model.predict_classes(neg))
fp_rate = float(fp)/neg.shape[0]

print("")
print("")

print("tp rate: ", tp_rate)
print("fp rate: ", fp_rate)