import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 				

from keras.preprocessing.sequence import pad_sequences
from numpy import argmax
from tensorflow_core.python.keras.models import load_model
from pickle import load

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import sys

#Integer to Word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

#Generate Caption
def generate_desc(model, tokenizer, photo, max_length):
	start = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([start])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)

		next_word = model.predict([photo,sequence], verbose=0)
		next_word = argmax(next_word)
		word = word_for_id(next_word, tokenizer)

		if word is None:
			break
		
		start += ' ' + word
		if word == 'endseq':
			break
	return start

model = load_model('model-ep003-loss3.638-val_loss3.869.h5')
tokenizer = load(open('tokenizer.pkl','rb'))

vgg = VGG16()
vgg._layers.pop()
vgg = Model(inputs=vgg.inputs,outputs=vgg.layers[-1].output)
image = load_img('../Webserver/assets/pic.jpg',target_size=(224,224))
image = img_to_array(image)
image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
image = preprocess_input(image)
photo = vgg.predict(image,verbose=0)


caption = generate_desc(model,tokenizer,photo,34)
caption = ' '.join(caption.split()[1:-1])
print(caption)
sys.stdout.flush()