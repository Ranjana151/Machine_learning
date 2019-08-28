import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']



plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Data processing
train_images=train_images/255.0
test_images=test_images/255.0


plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Training of model
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),loss="sparse_categorical_crossentropy",metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10)

test_loss,test_accuracy=model.evaluate(test_images,test_labels)
print("Accuracy is",test_accuracy)
predication=model.predict(test_images)
print(np.argmax(predication[0]))
plt.imshow(test_images[0])
plt.show()