from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from PIL import Image
import numpy as np

class classifier(object):
    """
    This class loads the convolutional neural net
    and handles all of the image processing and 
    prediction work for the given image. It also
    will save a copy of the image to the static/
    area after processing for viewing by the user
    """
    def __init__(self):
        """
        Builds the model with the same layer
        structure as the pre-training model.
        Then prepares the model for making predictions
        """
        self.width = 200
        self.height = 200

        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        self.model = Model(input=base_model.input, output=predictions)
        
        self.upload_pretrained_weights(self.model)
    
    def upload_pretrained_weights(self, model):
        """ 
        Adds the pretraining into the spawned model
        so that it knows about dogs and cats.
        ---
        Input: Conv. Neural Network
        """
        model.load_weights("first_try.h5")

    def predict(self, imgpath):
        """ 
        This pulls the image into the neural
        network and returns a prediction of
        dogness and catness of the image. 
        ---
        Input: path to image
        Output: Prediction of cat or dog
        """
        img_to_predict = np.empty(shape=(1,200,200,3))
        pic, mode, size = self.get_image(imgpath)
        img_to_predict[0] = np.reshape(pic,(200,200,3))
        prediction = self.model.predict(img_to_predict, batch_size=1, verbose=1)
        report = '{0}% Dog, {1}% Cat'.format(round(prediction[0][1]*100.,2),round(prediction[0][0]*100.,2))
        return report

    def get_image(self, image_path):
        """
        This method gets the image from the static
        area where the flask method for handling
        user input stores the image. It then modifies
        it and puts it into a format the neural network
        can work with. This includes converting
        between RGBA, P, and RGB encodings. It also
        resizes the image to 200x200
        ---
        Input: path to image
        output: image as array, mode of image, size of image
        """
        print "Loading the image..."
        image = Image.open(image_path, 'r')
        width, height = image.size
        imgr = image.resize((self.width,self.height),resample=Image.ANTIALIAS)
        imgr.save(image_path)
        
        if imgr.mode == 'RGB':
            pass
        elif imgr.mode in ['RGBA','P']:
            imgr = imgr.convert('RGB')
        else:
            print("Unsupported image mode: %s; Only JPG, PNG, and GIF modes are supported" % image.mode)
            return None
        
        pixel_values = list(imgr.getdata())
        pixel_values = np.array(pixel_values)/255.
        return pixel_values,imgr.mode,imgr.size
