from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import os.path

if not os.path.isfile("data/im.npy"):
    pos = np.loadtxt('data/im1.csv', delimiter=',', dtype=np.float32)
    # num_images = 77
    # IMAGE_SIZE = 2
    # NUM_CHANNELS = 2
    # pos=pos.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    np.save('data/im.npy', pos);
else:
    pos = np.load('data/im.npy')

if not os.path.isfile("data/cp.npy"):
    neg = np.loadtxt('data/cp1.csv', delimiter=',', dtype=np.float32)
    # num_images=143
    # IMAGE_SIZE=2
    # NUM_CHANNELS=2
    # neg=neg.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    np.save('data/cp.npy', neg);
else:
    neg = np.load('data/cp.npy')

pos_labels = np.ones((pos.shape[0], 1), dtype=int);
neg_labels = np.zeros((neg.shape[0], 1), dtype=int);

print("positive samples: ", pos.shape[0])
print("negative samples: ", neg.shape[0])

HIDDEN_LAYERS = 4

model = Sequential()

model.add(Dense(output_dim=HIDDEN_LAYERS, input_dim=pos.shape[1]))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.add(Activation("sigmoid"))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.vstack((pos, neg)), np.vstack((pos_labels, neg_labels)), nb_epoch=10, batch_size=128)

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