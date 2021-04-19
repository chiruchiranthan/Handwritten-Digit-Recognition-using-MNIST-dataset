import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings('ignore')
model = load_model('trained_model.h5')
image = cv2.imread('0.jpg')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    digit = thresh[y:y+h, x:x+w]
    resized_digit = cv2.resize(digit, (18,18))
    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
    prediction = model.predict(padded_digit.reshape(1, 28, 28, 1)) 
print("\n\n\n----------------Contoured Image--------------------")
plt.imshow(image, cmap="gray")
plt.show()
print ("\n\n---------------------------------------\n\n")
print ("=========PREDICTION============ \n\n")
plt.imshow(padded_digit.reshape(28, 28), cmap="gray")
plt.show()
print("\n\nPredicted output: {}".format(np.argmax(prediction)))













#model = load_model('trained_model.h5')
#image = cv2.imread('111.jpeg')
#grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
#contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#preprocessed_digits = []
#for c in contours:
#    x,y,w,h = cv2.boundingRect(c)
#    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
#    digit = thresh[y:y+h, x:x+w]
#    resized_digit = cv2.resize(digit, (18,18))
#    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
#    preprocessed_digits.append(padded_digit)
#print("\n\n\n----------------Contoured Image--------------------")
#plt.imshow(image, cmap="gray")
#plt.show()
#    
#inp = np.array(preprocessed_digits)
#
#for digit in preprocessed_digits:
#    prediction = model.predict(digit.reshape(1, 28, 28, 1))  
#    print ("\n\n---------------------------------------\n\n")
#    print ("=========PREDICTION============ \n\n")
#    plt.imshow(digit.reshape(28, 28), cmap="gray")
#    plt.show()
#    print("\n\nFinal Output: {}".format(np.argmax(prediction)))
##    
