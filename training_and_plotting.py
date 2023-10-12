import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense
import os
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')  # images from google drive
PATH="/content/drive/My Drive/dataset-resized/dataset-resized"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')
batch_size = 512
epochs = 10  
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) 
validation_image_generator = ImageDataGenerator(rescale=1./255) 
test_image_generator = ImageDataGenerator(rescale=1./255) 

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH), 
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

# base_model = tf.keras.applications.InceptionV3(include_top=False,
#     weights="imagenet",
#     input_shape=(150, 150, 3))
# base_model.trainable = False
# X = base_model.output
# X = Flatten(input_shape=base_model.output_shape[1:])(X)
# X = Dense(256, activation='relu')(X)
# X = Dropout(0.5)(X)
# Output = Dense(7, activation='softmax')(X)
# inception= Model(inputs=base_model.input, outputs=Output)

# inception.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#           optimizer=optimizers.Adam(lr = 0.001),
#           metrics=['accuracy'])

# history = inception.fit_generator(
#     train_data_gen,
#     epochs=epochs,
#     validation_data = val_data_gen)

# inception.save('/content/drive/My Drive/dataset-resized/training_inception_hdf5_format/trainedModel.h5')

#  base_model = tf.keras.applications.ResNet50V2(include_top=False,
#     weights="imagenet",
#     input_shape=(150, 150, 3))
 
# setup for resnet training from here 
base_model.trainable = False
X = base_model.output
X = Flatten(input_shape=base_model.output_shape[1:])(X)
X = Dense(256, activation='relu')(X)
X = Dropout(0.5)(X)
Output = Dense(7, activation='softmax')(X)
resnet= Model(inputs=base_model.input, outputs=Output)

resnet.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
          optimizer=optimizers.Adam(lr = 0.001),
          metrics=['accuracy'])

history = resnet.fit_generator(
    train_data_gen,
    epochs=epochs,
    validation_data = val_data_gen)


resnet.save('/content/drive/My Drive/dataset-resized/training_resnet_hdf5_format/trainedModelRN.h5')
# resnet model saved


# storing time it takes to predict for plotting
import time
# inception_time=[]
# for img in val_data_gen[0][0]:
#   img=np.expand_dims(img,axis=0)
#   start_time=time.time()
#   inception.predict(img)
#   inception_time.append(time.time()-start_time)

resnet_time=[]
for img in val_data_gen[0][0]:
  img=np.expand_dims(img,axis=0)
  start_time=time.time()
  resnet.predict(img)
  resnet_time.append(time.time()-start_time)


# testing  
import numpy
for imagePath in  test_data_gen[0][0]:
  imagePath=np.expand_dims(imagePath,axis=0)
  prediction = resnet.predict(imagePath)
  print(numpy.argmax(prediction))
  # 0 = bio
  # 1 = cardboard
  # 2 = glass
  # 3 = metal
  # 4 = paper
  # 5 = plastic
  # 6 = trash


# for plotting graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1,11)

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')


