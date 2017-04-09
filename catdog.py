
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from PIL import Image
import numpy as np
import os

class classifier(object):
    def __init__(self):
        self.width = 200
        self.height = 200
        base_model = InceptionV3(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # cat vs dog prediction
        predictions = Dense(2, activation='softmax')(x)

        # this is the model we will train
        self.model = Model(input=base_model.input, output=predictions)
        print "model created"
        self.upload_pretrained_weights(self.model)
        print "weights loaded"
    
    def upload_pretrained_weights(self, model):
        model.load_weights("first_try.h5")

    def prep_image(self, imgpath):
        img = Image.open(imgpath)
        imgr = img.resize((self.width,self.height),resample=Image.ANTIALIAS)
        img_data = np.array(imgr)/255.
        return img_data

    def predict(self, imgpath):
        print "predicting image"
        img_to_predict = np.empty(shape=(1,200,200,3))
        pic = self.prep_image(imgpath)
        img_to_predict[0] = pic
        prediction = self.model.predict(img_to_predict, batch_size=1, verbose=1)
        report = '{0}% Dog, {1}% Cat'.format(round(prediction[0][1]*100.,2), round(prediction[0][0]*100.,2))
        return report

    def save_image(self, fname):
        data = self.prep_image(fname)
        pic = Image.fromarray(data)
        pic.save('static/'+fname)
