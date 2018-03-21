'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function

from keras import callbacks
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2, 3, 5]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += ' '.join(sentence)
        print('----- Generating with seed: "' + ' '.join(sentence) + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars1)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence[maxlen-1]= next_char

            sys.stdout.write(' '+next_char)
            sys.stdout.flush()
        print()

if __name__ == '__main__':

    path ="C:\\Users\\Vivaswan Chandru\\Box Sync\\Mayanka-Research\\2018-CrisisCNN-WordEmbedding\\Code\\Keras_ImageClassification\\data\\IAPR_text"
    #path="D:\\Mayanka Lenevo F Drive\\Datasets\\Image_with_caption\\SBM\\SBU_captioned_photo_dataset_captions.txt"
    #path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
    print('corpus length:', len(text))

    #chars = sorted(list(set(text)))
    chars = list(text.split(" "))
    chars1=sorted(list(text.split(" ")))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars1))
    indices_char = dict((i, c) for i, c in enumerate(chars1))
    # char_indices = dict((c, i) for i, c in enumerate(chars))
    # indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 3
    step = 3
    sentences = []
    next_chars = []
    text=chars
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1


    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    tensorboard = callbacks.TensorBoard(
        log_dir='logs_LSTM',
        histogram_freq=0, batch_size=10, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
        embeddings_layer_names=None, embeddings_metadata=None)

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model.fit(x, y,
              batch_size=128,
              epochs=5,
              callbacks=[print_callback,tensorboard])