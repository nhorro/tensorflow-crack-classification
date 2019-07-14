# Crack classification with Tensorflow/Keras

This is part of project that attempts to reproduce the paper

## Instructions

### Project organization

```
./
	/data
	/doc
	/model-checkpoints
	/models
	/src
	/tensorboard_logs
	/training_logs
```

where:

- **data**: contains datasets and other media.
- **doc**: documentation files.
<<<<<<< Updated upstream
- **model-checkpoints**: checkpoints generated during model training in hd5 format.
=======
- **model-checkpoints**: checkpoints generated during model training in hd5f format.
>>>>>>> Stashed changes
- **models**: models converted to Tensorflow SavedModel format ready for deployment with tensorflow serving.
- **src**: notebooks and python scripts for model training and generating reports from training logs.
- **tensorboard_logs**: path to store tensorboard logs.
- **training_logs**: training logs in CSV to generate learning curves in reports.

### Train a model using docker image

**Note:**  the following instruction steps use a [custom docker image](https://hub.docker.com/r/nhorro/tensorflow1.12-py3-jupyter-opencv) based on official Tensorflow Docker image for GPU. 

Verifified with Ubuntu 18.04 and Geforce GTX 950M with nvidia driver version 390.116.

#### Steps

Clone repository

```bash
git clone https://github.com/nhorro/tensorflow-crack-classification.git
```

Download crack dataset

```bash
<<<<<<< Updated upstream
wget https://data.mendeley.com/datasets/5y9wdsg2zt/1/files/c0d86f9f-852e-4d00-bf45-9a0e24e3b932/Concrete%20Crack%20Images%20for%20Classification.rar
mkdir -pv data/datasets/cracks
unrar x Concrete\ Crack\ Images\ for\ Classification.rar ./data/datasets/cracks
```

Prepare a a training set and evaluation set.
=======
wget https://data.mendeley.com/datasets/5y9wdsg2zt/1/files/c0d86f9f-852e-4d00-bf45-9a0e24e3b932/Concrete%20Crack%20Images%20for%20Classification.rar?dl=1
```

Train the model
>>>>>>> Stashed changes

Export the model as Tensorflow SavedModel format to deploy with Tensorflow Serving.

### Serve a model using tensorflow-serving docker image

Official image tensorflow/serving:1.12.3-gpu is used for serving.

```bash
export SERVING_MODEL=febrero-cpu-friendly_weights
docker run -t --rm --runtime=nvidia -p 8501:8501 -v $(realpath $PWD/models):/models/ -e MODEL_NAME=$SERVING_MODEL tensorflow/serving:1.12.3-gpu
```
Query model status from:


http://localhost:8501/v1/models/febrero-cpu-friendly_weights

Query model metadata:


http://localhost:8501/v1/models/febrero-cpu-friendly_weights/metadata

<<<<<<< Updated upstream
## References

- 
=======
>>>>>>> Stashed changes
