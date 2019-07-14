import shutil 
import os
import tensorflow as tf

def convert_from_keras_to_savedmodel(input_filename, export_path):
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)
    model = tf.keras.models.load_model(input_filename)

    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    # Fetch the Keras session and save the model
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name:t for t in model.outputs})