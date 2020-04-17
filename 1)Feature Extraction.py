#Getting features of each image in the directory
from keras.applications.vgg16 import VGG16
from keras.models import Model
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from pickle import dump

def extract_features(directory):
    print('Getting 4096 element Vector representation of image:')

    #Loading the model 
    model = VGG16()

    #Removing the 16th layer of model
    model._layers.pop()
    model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
    
    #Description of our custom model
    print(model.summary())

    #Storing features of each photo in dictionary
    features = dict()
    for name in listdir(directory):
        filename = directory+'/'+name
        image = load_img(filename,target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image,verbose=0)

        imageid = name.split('.')[0]
        features[imageid] = feature
        print(imageid)
    return features

directory = "Demo_Dataset"
features = extract_features(directory)
print('Total images for feature extraction: ',len(features))
dump(features,open('img_features.pkl','wb'))