import tensorflow as tf
from matplotlib import pyplot
import warnings

warnings.filterwarnings('ignore')

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


for i in range(9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

num_of_trainImgs = x_train.shape[0] #60000 
num_of_testImgs = x_test.shape[0] #10000 
img_width = 28
img_height = 28
 
#reshapiing the training and testing data
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
input_shape = (img_height, img_width, 1)
 
#Normalizing the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train = tf.keras.utils.to_categorical(y_train, 10)


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))#rectified linear activation function
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)

model.save('trained_model.h5')
