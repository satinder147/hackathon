import cv2
import numpy as np
from dataloader import loader
from unet import Models
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
w=256
h=256
c=3


mod=Models(w,h,c)
auto_encoder=mod.arch3()
load_img=loader()

auto_encoder.summary()
x_data,y_data=load_img.load()
x_data=np.array(x_data,dtype='float')/255.0
y_data=np.array(y_data,dtype='float')/255.0
opt=Adam(lr=0.001,decay=0.001/50)
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.1,random_state=30)
auto_encoder.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
history=auto_encoder.fit(train_x,train_y,batch_size=1,shuffle='true',epochs=100,validation_data=(test_x,test_y),verbose=1)
auto_encoder.save('road3.MODEL')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("res.png")

