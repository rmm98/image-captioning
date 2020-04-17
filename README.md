# Image Captioning

![img](https://imgur.com/UvBPzjD.png)

**Caption Generation model developed using:**
* Python 3.7.3 with PIP 19.0.3 
* Keras 2.3.1 high level API on Tensorflow 2.1.0 backend 
* Dependencies:  numpy, scipy, matplotlib, pandas, scikit-learn, os, pickle
* Trained on Google Colab. [Interactive Python Notebook Link.](https://colab.research.google.com/drive/1nlJWffg8zt1ewsWmNDM6KuwoRZghRLZ-)
* Flickr8k Dataset

**Webserver developed using:**
* Javascript in NodeJS runtime environment
* Node modules: express, ejs, express-fileupload, fs, child_process

**Caption Generation model development & training:**
| Executable File (.py) | Output Artifact |
|--|--|
| Feature Extraction | img_features.pkl which is a dictionary of image features |
| Description | descriptions.txt which will contain cleaned descriptions |
| Load | tokenizer.pkl which has word-int mapping, I/O pairs to train the model |
| Define & Fit | model.h5 which is the defined model with minimum validation loss upon training |

**Final application executables:**
| Executable File | Purpose |
|--|--|
| Server.js | It listens to Client requests. Upon Image upload by Client, it spawns a python process to generate the caption and renders the final output to Client |
| Generate_Caption.py | Upon receiving the image, it uses model based on VGG16 to get the features. Using model.h5 and tokenizer.pkl, it generates the caption |

To save Caption Generation model development & training time, you can download all the output artifacts from [My Google Drive Link]()

**TO-DOs:**
* [ ] Train the model on Flick30k dataset instead of current Flickr8k dataset
* [ ] Add some Flair😎🎇✨ to the Webpage
