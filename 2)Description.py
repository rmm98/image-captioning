import string

#Opening the file n reading text
def load_description_document(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

#Creating dictionary containing ImageId and descriptions
def load_descriptions(doc):
	mapping = dict()

	for line in doc.split('\n'):
		tokens = line.split()
		if len(line) < 2:
			continue
		image_id, image_desc = tokens[0], tokens[1:]
		image_id = image_id.split('.')[0]
		image_desc = ' '.join(image_desc)
		if image_id not in mapping:
			mapping[image_id] = list()
		mapping[image_id].append(image_desc)

	return mapping

#Cleaning the description
def clean_descriptions(description):
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			desc = desc.split()
			#Lower case
			desc = [word.lower() for word in desc]
			#Remove Punctuation
			desc = [w.translate(table) for w in desc]
			#Remove a, s etc
			desc = [word for word in desc if len(word)>1]
			#Remove Num
			desc = [word for word in desc if word.isalpha()]
			
			desc_list[i] =  ' '.join(desc)

#Building Vocabulary
def to_vocabulary(descriptions):
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

#Saving final descriptions to file
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


filename = 'Demo_Text/Flickr8k.token.txt'
description_document = load_description_document(filename)

descriptions = load_descriptions(description_document)
print('Loaded: %d ' % len(descriptions))

clean_descriptions(descriptions)

vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

save_descriptions(descriptions, 'descriptions.txt')