#model submitted late with accuracy of 0.8967

CNN_model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1),padding='valid',strides=(1,1)),
    
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',padding='valid',strides=(1,1)),    
    layers.MaxPooling2D((2, 2),padding='valid',strides=None),
    layers.Dropout(0.25),

    
    #nick japmann adding 4th layer
    
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid',strides=(1,1)),
    layers.Dropout(0.25),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding='valid',strides=(1,1)),
    layers.MaxPooling2D((2, 2),padding='valid',strides=None),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
    
])
