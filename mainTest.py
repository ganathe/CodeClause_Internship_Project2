#NOTE
#if it predicts '0' then it is not affected by brain tumor
#if it predicts '1' then it is affected by brain tumor

import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.models import Sequential

model=load_model('BrainTumor10EpochsCategorical.h5')


#path of the image
image=cv2.imread('C:\\Users\\abcd\\OneDrive\\Desktop\\Brain_Tumor\\pred\\pred5.jpg')


img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img,axis=0)

result=model.predict(input_img)
predicted_classes=np.argmax(result,axis=1)
print(predicted_classes)

