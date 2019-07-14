#TODO: imports


def build_cracknet():
    """
        TODO: poner diferencias con paper e indicar donde se deben hacer
              cambios
    """
    # Convolution + Pooling #1
    model.add(Conv2D( 32, (11, 11), input_shape=(120,120,3),
                          activation = 'relu', use_bias=False ))    
    model.add(BatchNormalization())
    model.add(ELU())
    model.add( MaxPooling2D(pool_size = (7,7),strides=2))
    
    # Convolution + Pooling #2
    model.add(Conv2D( 48, (11, 11), activation = 'relu', 
                          use_bias=False ))    
    model.add(BatchNormalization())
    model.add(ELU())
    model.add( MaxPooling2D(pool_size = (5,5), strides=2))
    
    # Convolution + Pooling #3
    model.add(Conv2D( 64, (5, 5), activation = 'relu', use_bias=False ))    
         
    model.add(BatchNormalization())
    model.add(ELU())
    model.add( MaxPooling2D(pool_size = (3,3),strides=2))
    
    # Convolution + Pooling #4
    model.add(Conv2D( 80, (5, 5), activation = 'relu', use_bias=False ))    
    model.add(BatchNormalization())
    model.add(ELU())
    model.add( MaxPooling2D(pool_size = (3,3),strides=2))
    """   
    """
    # FC #1
    model.add( Dense( units = 5120, activation = 'relu' ) )
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add( Dense( units = 96, activation = 'softmax' ) )   
    
    # Compile
    # TODO: Ver, con binary cross_entropy no está dando la probabilidad de
    #       cada clase. Usar categorical_crossentropy o ver causa.
    sgd_optimizer = SGD(lr=0.002, decay=0.1/350, momentum=1)
    model.compile( optimizer = sgd_optimizer, 
                        loss = 'binary_crossentropy', 
                        metrics = ['accuracy'] )
    return model