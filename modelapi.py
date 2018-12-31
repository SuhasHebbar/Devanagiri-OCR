import numpy as np
import cv2
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D as Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
ocr=None
dir_path = os.path.dirname(os.path.realpath(__file__))
class ModelAPI:
    def __init__(self):
        self.data = []
        self.kernel = np.ones((5,5),np.uint8)

    def load_data(self):
        X = []#np.empty((0,9216), np.uint8)
        Y = np.zeros((1966,128), np.uint8)
        pathd ='~/preproc/'
        path = '~/preview/'


        i=0
        for filename in os.listdir(path):
            filepath = path + filename
              
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = self.preprocess(img,self.kernel)
            lst= filename.split('.')[0].split('_')
            lst.pop(0)
            lst.pop(0)
            lst.pop(0)
            print(i)
            for j in lst:
                Y[i][int(j) - 2304] = 1

            filepathd = pathd + filename
            #cv2.imwrite(filepathd, img)
            img = img.flatten().reshape([-1,9216])
                
              #print(X.shape)
            X.append(img)
              #print(X.shape)
            i+=1
        
        X= np.array(X)
        np.save('X.npy',X)
        np.save('Y.npy',Y)
        return X,Y
            
    def preprocess(self, img, kernel):
        img1 = cv2.fastNlMeansDenoising(img, 10, 7, 21)
        ret, img2 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
        openimg = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
        closeimg = cv2.morphologyEx(openimg, cv2.MORPH_CLOSE, kernel)
        rszimg = cv2.resize(closeimg, (96,96))
        ret, img3 = cv2.threshold(rszimg, 127, 255, cv2.THRESH_BINARY_INV)
        return img3
        
    def make_model(self):
        self.model = Sequential()
        
        self.model.add(Conv2D(32, kernel_size=(5,5), strides=(1, 1), activation='relu', input_shape=(96,96,1)))
        self.model.add(Conv2D(64, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(512, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='sigmoid'))
        
        self.model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001), metrics=[categorical_accuracy])

    def train(self):
        X=np.load('X.npy')
        Y=np.load('Y.npy')
        X = X.astype('float32')
        X /=255.0
        X = X.reshape(X.shape[0], 96, 96, 1)
        
        
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest')
                
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        X_test, X_validate, Y_test, Y_validate = train_test_split(X_test, Y_test, test_size=0.5)
        
        train_generator = train_datagen.flow(X_train, Y_train)  

        self.model.fit_generator(
                    train_generator,
                    steps_per_epoch=1000,
                    epochs=6,
                    validation_data=(X_validate,Y_validate))
                    
        score, acc = self.model.evaluate(X_test, Y_test, verbose=1)
        self.model.save('model.h5')
        print('score: ', score, 'accuraccy: ', acc)

    def predict(self, img):
        img1 = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img1 = self.preprocess(img1,self.kernel)
        
        img1 = img1.astype('float32')
        img1 = img1/255.0
        img1 = img1.reshape(1,96,96,1)
        
        preds = self.model.predict(img1)
        
        preds[preds >=0.5] = 1
        preds[preds < 0.5] = 0
        #print(preds.shape)
        #print((2304 + np.nonzero(preds)[1]).tolist())
        return (2304 + np.nonzero(preds)[1]).tolist()

    

def predict(img):
    global ocr
    #print("stuff")
    if ocr==None:
        ocr= ModelAPI()
        ocr.model = load_model(dir_path+'/model.h5')
	
    return ocr.predict(img)
