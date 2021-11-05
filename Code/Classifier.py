import import_ipynb # for google Colaboratory only
from DataLoader import labels, images_train, images_test
import numpy, visualkeras, csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Model Architecture
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()

# Model visualization
visualkeras.layered_view(model, legend= True) 

# Model compilation
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Model training
model.fit(images_train, labels,
          validation_split = 0.2,
          epochs = 10)

# Model evaluation
predictions = model.predict(images_test, 
                            verbose=1)

# CSV for submission
list_predicted = []
for value in predictions:
  list_predicted.append(int(numpy.where(value == numpy.amax(value))[0]))
list_predicted = numpy.array(list_predicted)
list_predicted = numpy.vstack(list_predicted)
Id = numpy.arange(1,len(list_predicted)+1)[:, numpy.newaxis]
data = numpy.column_stack((Id,list_predicted))
header = ['ImageId', 'Label']
with open('submission.csv','w', encoding='UTF8') as csv_predicted:
  writer = csv.writer(csv_predicted)
  writer.writerow(header)
  for row in data:
    writer.writerow(row)
    

'''
ACBC
'''
