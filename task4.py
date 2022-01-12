#! /usr/bin/python3

# Packages
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence

image_path = './food/{counter:05d}.jpg'
target_size = (224, 224) # (120, 176)

random_state = np.random.RandomState(0)
tf.random.set_seed(314)


# Functions
def shuffled_index(len_: int):
    _index = np.arange(len_)
    np.random.shuffle(_index)
    return _index


def load_food_img(c: int, target_size=target_size, as_array=True):
    _img = image.load_img(
        image_path.format(counter=c),
        target_size=target_size,
    )
    if as_array:
        _img = image.img_to_array(_img)
    return _img


class IMG_Seq(Sequence):
    
    def __init__(self, triplets: np.array, labels: np.array = None, batchsize=32, rnd_trans=False):
        self.triplets = triplets
        if type(labels) != type(None):
            self.labels = labels
        else:
            self.labels = [None] * self.triplets.shape[0]
        self.batchsize = batchsize
        self.rnd_trans = rnd_trans
        
        self.rnd_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=45, brightness_range=[.5, 1.5], shear_range=20, zoom_range=[.7, 1],
            channel_shift_range=100, fill_mode='nearest', horizontal_flip=True
        )
        
        self.pic_dict = np.zeros((10000, target_size[0], target_size[1], 3))
        for i in set(triplets.flatten()):
            self.pic_dict[i] = load_food_img(i)
        
    def __len__(self):
        return len(self.labels) // self.batchsize

    def __getitem__(self, idx):
        if self.rnd_trans:
            feats = [
                np.array([
                    self.rnd_img_gen.random_transform(self.pic_dict[i])
                    for i in self.triplets[idx*self.batchsize:(idx+1)*self.batchsize,j]
                ])
                for j in range(3)
            ]
        else:
            feats = [
                np.array([
                    self.pic_dict[i]
                    for i in self.triplets[idx*self.batchsize:(idx+1)*self.batchsize,j]
                ])
                for j in range(3)
            ]
        label = self.labels[idx*self.batchsize:(idx+1)*self.batchsize]
        return feats, label


base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    pooling='avg',
)

for i, layer in enumerate(base_model.layers):
    layer.trainable = False       
dense_layer = tf.keras.layers.Dense(8, activation='sigmoid')


def create_conv_layer(input):
    x = tf.keras.applications.mobilenet_v2.preprocess_input(input)
    x = base_model(x)
    x = dense_layer(x)
    x = tf.keras.Model(inputs=input, outputs=x,)
    return x


def distance(x):
    return tf.sqrt(tf.reduce_sum(tf.pow(x, 2), 1, keepdims=True))


# Load train data, split to train and validation data
train = np.loadtxt('./train_triplets.txt', dtype=int, delimiter=' ')

df = pd.DataFrame(train)
mod=5
fold_index = 1
train_index = df.index
df[(df[0]%mod!=fold_index) & (df[1]%mod!=fold_index) & (df[2]%mod!=fold_index)].index
val_index = df[(df[0]%mod==fold_index) & (df[1]%mod==fold_index) & (df[2]%mod==fold_index)].index

train_index_sets = np.concatenate((train[train_index], train[train_index][:, [0,2,1]]))
train_l = np.concatenate((np.ones_like(train_index), np.zeros_like(train_index)))

shuffled_train_index = shuffled_index(len(train_l))
train_index_sets = train_index_sets[shuffled_train_index]
train_l = train_l[shuffled_train_index]
    
val_index_sets = np.concatenate((train[val_index], train[val_index][:, [0,2,1]]))
val_l = np.concatenate((np.ones_like(val_index), np.zeros_like(val_index)))

seq_train = IMG_Seq(train_index_sets, train_l, rnd_trans=True)
seq_val = IMG_Seq(val_index_sets, val_l, batchsize=32,)

# The model
inputA = tf.keras.Input(shape=(target_size[0],target_size[1],3))
inputB = tf.keras.Input(shape=(target_size[0],target_size[1],3))
inputC = tf.keras.Input(shape=(target_size[0],target_size[1],3))

x = create_conv_layer(inputA)
y = create_conv_layer(inputB)
z = create_conv_layer(inputC)

AP = tf.keras.layers.Subtract(name='AP')([x.output, y.output])
AN = tf.keras.layers.Subtract(name='AN')([x.output, z.output])

d_AP = tf.keras.layers.Lambda(distance, name='d_AP')(AP)
d_AN = tf.keras.layers.Lambda(distance, name='d_AN')(AN)

combined = tf.keras.layers.Subtract()([d_AN, d_AP]) 
output = tf.keras.layers.Activation('sigmoid')(10. * combined)

model = tf.keras.Model(inputs=[x.input, y.input, z.input], outputs=output)

trainable_threshold = [156, 54]
epoch_list = [5, 30]
patience_list = [3, 5]
optimizer_list = [
    tf.keras.optimizers.Adam(),
    tf.keras.optimizers.SGD(lr=1e-4, momentum=0.8),
]

for r in range(len(trainable_threshold)):
    for i, l in enumerate(model.layers[9].layers):
        if i < trainable_threshold[r]:
            l.trainable = False
        else:
            l.trainable = True

    model.compile(
        optimizer=optimizer_list[r],
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    model.summary()

    checkpoint_path = './triplet_loss/cp_{counter:d}.ckpt'.format(counter=r)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
    )
    es_callback = tf.keras.callbacks.EarlyStopping(
    	monitor='val_loss', patience=patience_list[r], mode='min'
    )

    if r == 0:
        initial_epoch = 0
    else:
        initial_epoch = history.epoch[-1]+1

    history = model.fit(
        seq_train,    
        epochs=epoch_list[r],
        initial_epoch=initial_epoch,
        validation_data=seq_val,
        callbacks=[cp_callback,],
        verbose=2,
    )
    model.save_weights('./checkpoints/triplet_loss_{counter:d}'.format(counter=r))

    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    if os.path.exists('./history.csv'):
        history_df.to_csv('./history.csv', header=False, index=False, mode='a')
    else:
        history_df.to_csv('./history.csv', header=True, index=False, mode='a')

# Prediction and export
print('Start with prediction')
test = np.loadtxt('./test_triplets.txt', dtype=int, delimiter=' ')
pred = [model.predict(param[0])[0,0] > .5 for i, param in enumerate(IMG_Seq(test, batchsize=1))]
print('Compare length of test set: ',len(pred)==test.shape[0])
np.savetxt('submission.txt', pred, fmt='%d')
print('Job is done.')
