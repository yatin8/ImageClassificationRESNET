import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense,Activation,Flatten,Input,Dropout,GlobalAveragePooling2D





base_path = '/home/yatin/yatin/projects/CNN/Dataset/'
dirs = os.listdir('Dataset/')
print(dirs)
folder_path = ""
image_data = []
labels = []
label_dict = {'cats':0,'dogs':1,'horses':2,'humans':3}
for ix in dirs:
    path = os.path.join(base_path,ix)
    img_data = os.listdir(path)
    for im in img_data:
        img = image.load_img(os.path.join(path,im),target_size=(224,224))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(label_dict[ix])


print(len(image_data), len(labels))
combined = list(zip(image_data, labels))
random.shuffle(combined)

image_data[:], labels[:] = zip(*combined)
X_train = np.array(image_data)
Y_train = np.array(labels)
print(X_train.shape,Y_train.shape)

Y_train = np_utils.to_categorical(Y_train)
print(X_train.shape,Y_train.shape)



model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3))
model.summary()

av1 = GlobalAveragePooling2D()(model.output)
fc1 = Dense(256,activation='relu')(av1)
d1 = Dropout(0.5)(fc1)
fc2 = Dense(4,activation='softmax')(d1)

model_new = Model(inputs=model.input, outputs= fc2)
model_new.summary()

adam = Adam(lr=0.00003)
model_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

for ix in range(len(model_new.layers)):

    print(ix, model_new.layers[ix])

for ix in range(169):
    model_new.layers[ix].trainable = False
    
model_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model_new.summary()

hist = model_new.fit(X_train,Y_train,
                    shuffle = True,
                    batch_size = 16,
                    epochs = 5,
                    validation_split=0.20
                    )


plt.figure(0)
plt.plot(hist.history['acc'],'b')
plt.plot(hist.history['val_acc'],'g')
plt.plot(hist.history['loss'],'black')
plt.plot(hist.history['val_loss'],'red')
plt.show()