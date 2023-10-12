#from gpiozero import Servo
import RPi.GPIO as GPIO
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense
import os
import numpy as np

PATH="/home/pi"
test_dir = os.path.join(PATH, 'test_img_store')

batch_size = 512
epochs = 10  #changed from 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

test_image_generator = ImageDataGenerator(rescale=1./255)


resnet = tf.keras.models.load_model('/home/pi/projects/training_resnet_hdf5_format/trainedModelRN.h5')



import pygame
import pygame.camera

width = 1920
height = 1080

pygame.init()
pygame.camera.init()
camlist = pygame.camera.list_cameras()
cam = pygame.camera.Camera(camlist[0],(width,height))

nn = int(1)
if nn == 1:
  cam.start()
  image = cam.get_image()
  time.sleep(2)
  cam.stop()
  pygame.image.save(image,'/home/pi/test_img_store/images_dir/image.jpg')
  test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size, directory=test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')
  # imagePath = '/home/pi/test_img_store/images_dir/image.jpg'
  imagePath = test_data_gen[0][0][0]
  imagePath=np.expand_dims(imagePath,axis=0)
  prediction = resnet.predict(imagePath)
  #print(np.argmax(prediction))
  # 6 = trash
  # 5 = plastic
  # 0 = bio
  # 1 = cardboard
  # 4 = paper
  # 3 = metal
  # 2 = glass
  if np.argmax(prediction)== 4 or np.argmax(prediction)== 1 or np.argmax(prediction)== 0:
    print('biodegradable')
    GPIO.setmode(GPIO.BCM)  # set up BCM GPIO numbering
    servo = 18
    GPIO.setup(servo, GPIO.OUT)
    p=GPIO.PWM(servo,50)
    p.start(7.5)
    print(1)
    time.sleep(1)
    p.ChangeDutyCycle(10.5)
    time.sleep(5)
    p.ChangeDutyCycle(7.5)
    time.sleep(1)
  else:
    GPIO.setmode(GPIO.BCM)  # set up BCM GPIO numbering
    servo = 18
    GPIO.setup(servo, GPIO.OUT)
    p=GPIO.PWM(servo,50)
    p.start(7.5)

    print(0)
    time.sleep(1)
    p.ChangeDutyCycle(4.5)
    time.sleep(5)
    p.ChangeDutyCycle(7.5)
    time.sleep(1)
    print('Non-biodegradable')