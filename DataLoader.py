import pandas,numpy
from tensorflow.keras.utils import to_categorical


# Dataset loader
load_train = pandas.read_csv("digit-recognizer/train.csv")   
load_test = pandas.read_csv("digit-recognizer/test.csv")   

# Image container creation
images_train = []
images_test = []

# Dataset processing
    # train Data
for row in range(0,len(load_train)):
    images_arr = numpy.zeros((28, 28, 1))
    pixelx = 0
    pixely = 0
    
    if (row % 1000) == 0:
        print("Loading training images: ",row,"/",len(load_train),
              str(round((row/len(load_train))*100))+"%")
        
    for column in load_train.columns:
        
        if column == 'label':
            pass
        
        else:
            images_arr[pixely][pixelx][0] = load_train.loc[row,column]
            pixelx += 1
            
            if pixelx == 28:
                pixelx = 0
                pixely += 1 
                
    images_train.append(images_arr)

    # test data
for row in range(0,len(load_test)):
    images_arr = numpy.zeros((28, 28, 1))
    pixelx = 0
    pixely = 0
    
    if (row % 1000) == 0:
        print("Loading testing images: ",row,"/",len(load_test), 
              str(round((row/len(load_test))*100))+"%")
        
    for column in load_test.columns:
    
        images_arr[pixely][pixelx][0] = load_test.loc[row,column]
        pixelx += 1
        
        if pixelx == 28:
            pixelx = 0
            pixely += 1 
            
    images_test.append(images_arr)

# Array conversion and Standardization
images_train = numpy.array(images_train).astype("float32") / 255
images_test = numpy.array(images_test).astype("float32") / 255


# Label acquisition
labels = to_categorical(load_train.label)

'''
ACBC
'''


