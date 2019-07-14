# -*- coding: utf-8 -*-
# TODO: imports


def build_simplenet(input_shape=(64,64,3),n_output_classes=2):
    """
    """
    model = Sequential()
    
    # Convolution + Pooling #1
    model.add(Conv2D( 32, (3, 3), input_shape=input_shape,
                          activation = 'relu' ))        
    model.add( MaxPooling2D(pool_size = (2,2)))
    
    # Convolution + Pooling #2
    model.add(Conv2D( 32, (3, 3), activation = 'relu' ))        
    model.add( MaxPooling2D(pool_size = (2,2)))
    
    # Flattening
    model.add( Flatten() )
    
    # FC #1
    model.add( Dense( units = 128, activation = 'relu' ) )
    
    # Output Layer
    model.add( Dense( units = n_output_classes, activation = 'softmax' ) )   
    
    # Compile
    model.compile( 
        optimizer = 'adam', loss = 'categorical_crossentropy',
        metrics = ['accuracy'] )
    return model



def train_simplenet( model,
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
    return True
