from imgin import ImageHelper
from autoencoder import AAE
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.utils import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
from sklearn.model_selection import train_test_split

# input image dimensions
img_rows, img_cols = 512, 512

# number of channels
img_channels = 1

#%%

path1 = 'assets'    #path of folder of images    
path2 = 'upd'  #path of folder to save images    

listing = os.listdir(path1)
num_samples=size(listing)
print(num_samples)

for file in listing:
    print(file)
    im = Image.open(path1 + '/' + file)  
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(path2 +'/' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open('upd' + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('upd'+ '\\' + im2)).flatten()
              for im2 in imlist],'f')

label=np.ones((num_samples,),dtype = int)
label[0:89]=0
label[89:187]=1
label[187:]=2


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img=immatrix[14].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)




(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

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
aae.train(125, X_train, batch_size=32)
