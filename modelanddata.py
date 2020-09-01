from tensorflow.keras.datasets import fashion_mnist
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import ssl
from tensorflow.keras.utils import to_categorical

requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

y_cat_test = to_categorical(y_test,num_classes=10)
y_cat_train = to_categorical(y_train,num_classes=10)

x_train = x_train/255 #Dividimos los valores entre 255 para solo tener valores entre 0 y 1 en la escala de grises de la imagen
x_test = x_test/255
x_train = x_train.reshape(60000,28,28,1) #Se especifica que solo es una escala de colores
x_test = x_test.reshape(10000,28,28,1)
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout,Flatten
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=5,activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))

#Output Layer
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=1)

model.fit(x_train,y_cat_train,epochs=30,validation_data=(x_test,y_cat_test),callbacks=[early_stop])
model.evaluate(x_test,y_cat_test,verbose=0)

from sklearn.metrics import classification_report,confusion_matrix
pred = np.argmax(model.predict(x_test), axis=-1)
print(classification_report(y_test,pred))
print('\n')

print(confusion_matrix(y_test,pred))
print('\n')
print(np.argmax(model.predict(x_test[10].reshape(1,28,28,1))),axis=-1) #Prueba de predicción
plt.imshow(x_test[10],cmap='Greys')
