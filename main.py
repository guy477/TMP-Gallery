from imgin import ImageHelper
from autoencoder import AAE
import numpy as np
import pandas as pd
import wget
#from tensorflow.keras.datasets import fashion_mnist
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
import matplotlib
#import theano
import FaceCrop as fc
from PIL import Image
from numpy import *
from sklearn.model_selection import train_test_split

# input image dimensions
img_rows, img_cols = 512, 512

# number of channels
img_channels = 1

#%%

path1 = 'assets/Faces'    #path of folder of images    
path2 = 'upd/Facess'  #path of folder to save images    

listing = os.listdir(path1)
num_samples=size(listing)
print(num_samples)
"""
dat = pd.read_csv("assets/faceexp-comparison-data-train-public.csv", error_bad_lines=False)
print(dat[dat.columns[5]])
urls = dat[dat.columns[5]]"""
"""
ind = 0
for url in urls:
    print(url)
    try:
        wget.download(url, "assets/Faces/face{}.jpg".format(ind))
        ind+=1
    except:
        pass
    if(ind == 500):
        break

for file in listing:
    print(file)
    #im = Image.open(path1 + '/' + file)  
    gray = fc.facecrop(path1 + '/' + file)
    try:
        img = Image.open(path2+'/'+file).resize((img_rows,img_cols))
        gray = img.convert('L')
                    #need to do some more processing here          
        gray.save(path2 +'/' +  file, "JPEG")
    except:
        print("no face :(")

"""
imlist = os.listdir(path2)

im1 = array(Image.open('upd/Facess' + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images
print(imnbr)
# create matrix to store all flattened images
immatrix = array([array(Image.open('upd\\Facess'+ '\\' + im2)).flatten()
              for im2 in imlist],'f')

label=np.ones((imnbr,),dtype = int)
#a = int(len(imlist)/3)
label[:]=0
#label[a:2*a]=1
#label[2*a:]=2


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]


(X, y) = (train_data[0],train_data[1])


# split into training and testing sets (I will have no test data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=4)


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

#convert pixel value to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#compress to help with training time
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
#X_train.shape = tuple(reversed(X_train.shape))
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_test[0].shape)



"""
(X, _), (_, _) = fashion_mnist.load_data()
X_train = X / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)
print(X_train[0].shape)
"""

image_helper = ImageHelper()
aae = AAE(X_train[0].shape, image_helper)
aae.train(10000, X_train, batch_size=32)
