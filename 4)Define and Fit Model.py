from LoadingDataset_Encoding_IOpairs import X1train,X2train,ytrain,max_length,vocab_size
from LoadingDataset_Encoding_IOpairs import X1test,X2test,ytest

from keras.layers import Input,Dropout,Dense,Embedding,LSTM
from keras.layers.merge import add
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

#Captioning model
def define_model(vocab_size, max_length):

	#4096 -> 256 element vector represenation of image
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)

	#Word embedding and LSTM
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)

	#Decoder
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)

	#Tieing it together -> [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

model = define_model(vocab_size,max_length)

#Checkpoint
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#Fit
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))