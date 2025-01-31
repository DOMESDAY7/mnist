# All For One

### First model 

The first was to do a simple model with just Dense layers and `0.001` of learning rate. In fact it works well the model reached `97.40%` of accuracy during the test

```py
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

### Second model

The second model will be, like all others after it, an iteration of the previous one. Here we just add a Conv2D layer wich is a convolutional that is very performant when classify images.

```py
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

Also we reduce the learning rate to `0.0001`. In order to learn maybe slowly but deeply. With this type of model we reached `98.50%` during the test.