from pickle import load,dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array

#Picking Test Dataset
def load_doc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

def load_testid(filename):
    testid_doc = load_doc(filename)
    dataset = list()
    for line in testid_doc.split('\n'):
        if len(line)<1:
            continue
        dataset.append(line.split('.')[0])
    return set(dataset)

#Loading Dataset images & desc
def load_clean_description(filename,dataset):
    clean_descriptions_doc = load_doc(filename)
    clean_descriptions = dict()

    for line in clean_descriptions_doc.split('\n'):
        tokens = line.split()
        image_id,image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in clean_descriptions:
                clean_descriptions[image_id] = list()
            desc = 'startseq'+' '.join(image_desc)+'endseq'
            clean_descriptions[image_id].append(desc)
    
    return clean_descriptions

def load_img_features(filename,dataset):
    all_features = load(open(filename,'rb'))
    features = {key : all_features[key] for key in dataset}
    return features

#Encoding
def to_listOfLines(clean_descriptions):
	all_desc = list()
	for key in clean_descriptions.keys():
		[all_desc.append(d) for d in clean_descriptions[key]]
	return all_desc

def create_tokenizer(clean_descriptions):
	lines = to_listOfLines(clean_descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

#Input Output pairs for Training
def create_sequences(tokenizer, descriptions, photos, max_length, vocab_size):
	X1, X2, y = list(), list(), list()
	
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			seq = tokenizer.texts_to_sequences([desc])[0]
			for i in range(1, len(seq)):
				in_seq, out_seq = seq[:i], seq[i]
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)

	return array(X1), array(X2), array(y)

#Max length of description
def max_length(descriptions):
	lines = to_listOfLines(descriptions)
	return max(len(d.split()) for d in lines)

filename = 'Demo_Text/Flickr_8k.trainImages.txt'
train = load_testid(filename)
print('Train dataset : %d' % len(train))

train_descriptions = load_clean_description('descriptions.txt',train)
train_images = load_img_features('img_features.pkl',train)

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
dump(tokenizer,open('tokenizer.pkl','wb'))

max_length = max_length(train_descriptions)
print(max_length)

X1train,X2train,ytrain = create_sequences(tokenizer,train_descriptions,train_images,max_length,vocab_size)

#To validate durng training
filename = 'Demo_Text/Flickr_8k.devImages.txt'
test = load_testid(filename)
print('Test dataset : %d' % len(test))

test_descriptions = load_clean_description('descriptions.txt',test)
test_images = load_img_features('img_features.pkl',test)

X1test,X2test,ytest = create_sequences(tokenizer,test_descriptions,test_images,max_length,vocab_size)