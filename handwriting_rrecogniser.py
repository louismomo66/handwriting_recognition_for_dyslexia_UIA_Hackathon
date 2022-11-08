# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(512, (3, 3), input_shape = (32,32,3), activation = 'relu'))
classifier.add(Conv2D(512, (3, 3), activation = 'relu'))


# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\Louis\Desktop\mak\Gambo\Train',
                                                 target_size = (32, 32),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\Louis\Desktop\mak\Gambo\Test',
                                            target_size = (32, 32),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit(training_set,
                         steps_per_epoch = 2000,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('model_4.h5')#saving the model

# Part 3 - Making new predictions

import numpy as np
# from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array 
test_image = load_img(r'C:\Users\Louis\Desktop\mak\Gambo\Test\Corrected\4_3.png', target_size = (32,32))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    print('Corrected')
elif result[0][1] == 1:
    print('Normal')
elif result[0][2] == 1:
    print('Reversal')
else:
    print("Nothing")
