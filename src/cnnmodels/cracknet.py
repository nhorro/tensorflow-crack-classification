# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ELU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

from IPython.display import SVG
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def build_cracknet(input_shape=(120,120,3),n_output_classes=2):    
    model = Sequential()
    
    bias_regularization = 0.0001
    kernel_regularization = 0.0001
    
    # Convolution + Batch Norm. + ELU + Pooling #1  
    model.add( 
        Conv2D( 32, (11, 11), 
               input_shape=input_shape, 
               activation = 'relu', 
               strides=(1,1), 
               name="Conv1",
               kernel_regularizer=l2(kernel_regularization),
               bias_regularizer=l2(bias_regularization)
        ) ) 
    model.add( BatchNormalization( name="Conv1BN"))
    #model.add( ELU( name="Conv1ELU" ))
    model.add( MaxPooling2D(pool_size = (7,7),strides=(2,2), name="Conv1Pool" ) )
    
    # Convolution + Batch Norm. + ELU + Pooling #2
    model.add( 
        Conv2D( 48, (11, 11), 
               input_shape=(52,52,32), 
               activation = 'relu', 
               strides=(1,1), name="Conv2",
               kernel_regularizer=l2(kernel_regularization),
               bias_regularizer=l2(bias_regularization)
        ) )
    model.add( BatchNormalization( name="Conv2BN" ))
    #model.add( ELU( name="Conv2ELU" ))
    model.add( MaxPooling2D(pool_size = (5,5),strides=(2,2), name="Conv2Pool" ) )
    
    # Convolution + Batch Norm. + ELU + Pooling #3
    model.add(
        Conv2D( 64, (7,7), 
        input_shape=(19,19,48), 
        activation = 'relu', 
        strides=(1,1), 
        name="Conv3",
        kernel_regularizer=l2(kernel_regularization),
        bias_regularizer=l2(bias_regularization)
    ) )
    model.add(BatchNormalization(name="Conv3BN"))
    #model.add(ELU( name="Conv3ELU"))
    model.add( MaxPooling2D(pool_size = (3,3),strides=(1,1), name="Conv3Pool" ) ) 
    
    # Convolution + Batch Norm. + ELU + Pooling #4
    model.add(
        Conv2D( 80, (5,5), 
               activation = 'relu', 
               strides=(1,1), 
               name="Conv4",
               kernel_regularizer=l2(kernel_regularization),
               bias_regularizer=l2(bias_regularization)
        ))
    model.add(BatchNormalization( name="Conv4BN" ))
    #model.add(ELU( name="Conv4ELU" ))
    model.add( MaxPooling2D(pool_size = (3,3),strides=(1,1), name="Conv4Pool" ) ) # Paper says (2,2) ?
    
    # Flattening
    model.add( Flatten() )
    
    # FC #1
    model.add( Dense( units = 5120, input_shape=(96,), activation = 'relu', ) )
    model.add(ELU())
    model.add(Dropout(0.2))
    
    # FC #2
    model.add( Dense( units = 96, input_shape=(2,), activation = 'relu' ) )   
    
    # Output Layer
    model.add( Dense( units = n_output_classes, activation = 'softmax' ) )   
    
    # Compile
    sgd_optimizer = SGD(lr=0.002, decay=0.1/350, momentum=1)
    model.compile( optimizer = sgd_optimizer, 
                        loss = 'categorical_crossentropy', 
                        metrics = ['accuracy'] )
    return model




def train_cracknet( model,
               target_size,
               dataset_path,
               training_path_prefix,
               test_path_prefix,                        
               history_file_path,
               history_filename,
               checkpoint_path,
               checkpoint_prefix,
               number_of_epochs,
               tensorboard_log_path
            ):
    """
        see: https://keras.io/preprocessing/image/
    """
    train_datagen = ImageDataGenerator( rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True )
        
    test_datagen = ImageDataGenerator(rescale=1./255)
    training_set_generator = train_datagen.flow_from_directory(
            dataset_path+training_path_prefix,
            target_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
    test_set_generator = test_datagen.flow_from_directory(
            dataset_path+test_path_prefix,
            target_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
    step_size_train=training_set_generator.n//training_set_generator.batch_size
    step_size_validation=test_set_generator.n//test_set_generator.batch_size

    check_pointer = ModelCheckpoint(
            checkpoint_path + '%s_weights.{epoch:02d}-{val_loss:.2f}.hdf5' % checkpoint_prefix, 
            monitor='val_loss', 
            mode='auto', 
            save_best_only=True
    )
    
    tensorboard_logger = TensorBoard( 
        log_dir=tensorboard_log_path, histogram_freq=0,  
          write_graph=True, write_images=True
    )
    tensorboard_logger.set_model(model)

    csv_logger = CSVLogger(filename=history_file_path+history_filename)
    history = model.fit_generator(
            training_set_generator,
            steps_per_epoch=step_size_train,
            epochs=number_of_epochs,
            validation_data=test_set_generator,
            validation_steps=step_size_validation,
            callbacks=[check_pointer, csv_logger,tensorboard_logger] 
    )