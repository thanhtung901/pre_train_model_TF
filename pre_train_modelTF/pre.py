import tensorflow 
from tensorflow.keras.applications.inception_v3 import InceptionV3

model = InceptionV3(
    input_shape = (150,150,3),
    include_top=False,
    weights=None,
)
for layer in model.layers:
    layer.trainable = False
last_layer = model.get_layer('mixed10')
out_put_last = last_layer.output
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop


x = Flatten()(out_put_last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation = 'sigmoid')(x)

from tensorflow.keras.models import Model
models = Model(model.input, x)
models.summary()