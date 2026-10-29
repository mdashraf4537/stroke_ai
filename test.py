import tensorflow as tf
m = tf.keras.models.load_model("models/stroke_cnn_model_v2.keras", compile=False)
print(m.input_shape)  # e.g. (None,224,224,3) -> IMG_SIZE
print(m.output_shape) # e.g. (None,3) -> num classes
for l in m.layers[-5:]: print(l.name)