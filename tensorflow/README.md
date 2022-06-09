# Custom Training Loop in Tensorflow

## Defining Model
There are many way to define the model in tensorflow.

### Option 1
Using ```Sequential``` to chain layers sequentially.
```python
model = Sequential(
    [
        layers.Conv2D(
            32,
            3,
            padding="same",
            activation="relu",
            input_shape=(im_size, im_size, 3),
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.Dense(1),
    ]
)
```

or initial ```Sequential``` and then, add another layers by using ```.add()```

```python
model = Sequential()
model.add(layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        input_shape=(im_size, im_size, 3)
    ))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1))
```

### Option 2
Using the Functional API to implementing complex models.
```python
input_layer = Input(shape=(im_size, im_size, 3),)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(input_layer)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
output_layer = layers.Dense(1)(x)

model = Model(inputs=input_layer, outputs=[output_layer])
```

### Option 3
Implement model as a Class

Inheriting from the existing Model class lets you use the ```Model``` methods such as ```compile()```, ```fit()```, ```evaluate()```.  
When inheriting from ```Model```, you will want to define at least two functions:  

```__init__()```: you will initialize the instance attributes.  
```call()```: you will build the network and return the output layers.

```python
class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.pool = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.conv3 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv4 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.fc1 = layers.Dense(512, activation="relu")
        self.fc2 = layers.Dense(1)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
model = CustomModel()
```

## Custom loss
We have to define a function that accepts y_true and y_pred as parameters. We then compute and return the loss value.
```python
def custom_loss(y_true, y_pred):
  // do something
  return {your_loss}
```


## Training the model
Tensorflow can easily compile and train model by using built-in function.
```python
model.compile(...)
model.fit(...)
```
### Custom training loop
GradientTape is required for custom training loop.
```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    acc_value = tf.math.equal(
        y, tf.math.round(tf.keras.activations.sigmoid(logits))
    )
    train_acc.update_state(acc_value)
    train_loss.update_state(loss_value)
```
You can use the ```@tf.function``` decorator to automatically generate the graph-style code. This will help you create performant and portable models  
Using radient tape to calculate the gradients and then update the model trainable weights using the optimizer.
