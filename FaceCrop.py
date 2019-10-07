import os
import cv2

def facecrop(image):
    fa = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(fa)
    img = cv2.imread(image)

    m = (img.shape[1],img.shape[0])
    minimg = cv2.resize(img, m)

    faces = cascade.detectMultiScale(minimg)
    
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        #print("upd/Facess")
        
        cv2.imwrite("upd/Facess"+image[image.index("/"):][image.index('/'):], sub_face)
    pass
