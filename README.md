# 	Crack classification with Tensorflow/Keras

This is repository is part of [the nuclear plant crack inspection processing pipeline](https://github.com/nhorro/nuclear-plant-crack-inspection-pipeline) that attempts to reproduce the paper *Deep Learning-based Crack Detection Using Convolutional Neural Network and Naıve Bayes Data Fusion.* [1] 

Two CNN models for crack detection are implemented in Tensorflow/Keras. A very simple model to test the workflow on low-end computers (that we call SimpleNet) and the model from the original paper (which we call CrackNet) is implemented in Tensorflow and Keras.

Also, transfer learning is applied to the VGG-16 model.

Any of these models is then embedded in the CNNDetector component that takes an image as an input, scans it for cracks, and returns a list of the bounding boxes with the probabilities of each being a crack. The *CNNDetector* in the following nuclear plant inspection processing pipeline can be implemented by locally instancing the tensorflow executor or consuming the model as as service with REST or GRPC.

![Nuclear Plant Inspection Processing Pipeline](doc/assets/nuclear-plant-crack-inspection-pipeline-1.png)



## Overview of datasets and CNN architectures

The provided pre-trained models are the result of experimenting with different dataset preparations and model architectures and hyperparameter selection.

### Datasets

The following are publicly available datasets for crack and surface defect classification.  A script is provided to download and prepare first dataset (instructions below).

| Dataset                                                      | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Concrete Crack Images for Classification](http://dx.doi.org/10.17632/5y9wdsg2zt.1) | Croncrete images having cracks collected from various buildings separated in two classes: "Positive crack" and "Negative crack". Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels. |
| [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) | SDNET2018 is an annotated image dataset for training, validation, and benchmarking of artificial intelligence based crack detection algorithms for concrete. SDNET2018 contains over 56,000 images of cracked and non-cracked concrete bridge decks, walls, and pavements. The dataset includes cracks as narrow as 0.06 mm and as wide as 25 mm. |
| [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) | In the Northeastern University (NEU) surface defect database, six kinds of typical surface defects of the hot-rolled steel strip are collected, i.e., rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). The database includes 1,800 grayscale images: 300 samples each of six different kinds of typical surface defects. |
| [Magentic tile defect dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.) | This is the datasets of the upcoming paper "Saliency of magnetic tile surface defects"The images of 6 common magnetic tile defects were collected, and their pixel level ground-truth were labeled.  (From original author description). |

#### Dataset augmentation

Two notebooks to experiment with [basic augmentation](src/notebooks/Image Augmentation - Basic.ipynb) and [PCA color augmentation](src/notebooks/Image Augmentation - Color PCA.ipynb) are provided.

#### Dataset directory structure

Datasets should be splitted in two or three sets (training and validation or training,validation, and testing set) and each set should a directory with the name of the class contain images of that class. This is the default directory structure used by Keras [flow_from_directory](https://keras.io/preprocessing/image/) method.

```
./data
	datasets
		training_set
			crack
				imgxyz.jpg
				imgxyz.jpg
				...
			no_crack
				imgxyz.jpg
				imgxyz.jpg
				...
		validation_set
			crack
				imgxyz.jpg
				imgxyz.jpg
				...
			no_crack
				imgxyz.jpg
				imgxyz.jpg
				...
		test_set
			crack
				imgxyz.jpg
				imgxyz.jpg
				...
			no_crack
				imgxyz.jpg
				imgxyz.jpg
				...
```

### CNN models

The following CNN model architectures are provided:

| Model                              | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| [SimpleNet(Keras)](src/notebooks/) | Very simple CNN. CPU friendly.                               |
| [CrackNet(Keras)]()                | Implementation of CNN described in [1] in Keras with Tensorflow backend. |
| [CrackNet(Tensorflow)]()           | Implementation of CNN described in [1] in Tensorflow.        |
| [VGG16]()                          | Transfer learning applied to VGG16.                          |



## Pre-trained models

#### SimpleNET trained on cracks dataset

SimpleNET was trained on a public available cracks in concrete surface dataset [2] to be used as the model in the CNN detector for the nuclear plant crack inspection pipeline.

##### Training parameters and results

Original dataset consists of 20.000 samples for each class. For training session 18.000 samples were used for training and 2.000 samples were reserved for validation.

Some data augmentation was performed using Keras ImageDataGenerator ( see: https://keras.io/preprocessing/image/).

| Parameter                 | Value                     |
| ------------------------- | ------------------------- |
| Optimizer                 | ADAM                      |
| Loss function             | Categorical Cross Entropy |
| Epochs                    | 30                        |
| Batch size                | 32                        |
| Data augmentation rescale | 1./255                    |
| Shear range               | 0.2                       |
| Zoom range                | 0.2                       |
| Horizontal flip           | True                      |

###### Learning curves

Accuracy of near 98% was obtained against test set for checkpoint file: simplenet_cracks_weights.29-0.01.hdf5.

![simplenet-cracks-trainingreport](doc/assets/simplenet-cracks-trainingreport.png)

Testing against a real high res image containing cracks, the model fails to detect some positives and it is evident that more negative examples are needed for scenarios when surface contains elevations or other variations (bottom left section):

![crack_detections](doc/assets/crack_detections.jpeg)

### CrackNET trained on cracks dataset

WIP

## Instructions

### Project organization

```
./
	/data
	/doc
	/model-checkpoints
	/models
	/src
		notebooks
			data_preparation
			cnn_model_development
			examples
		utils
	/tensorboard_logs
	/training_logs
```

where:

- **data**: contains datasets and other media.
- **doc**: documentation files.
- **model-checkpoints**: checkpoints generated during model training in hd5 format.
- **models**: models converted to Tensorflow SavedModel format ready for deployment with tensorflow serving.
- **src**: source code directory containing:
  - notebooks:
    - **data_preparation**: notebooks for EDA (Exploratory Data Analysis) and experiment with dataset augmentation.
    - **cnn_model_development**: notebooks for development and training of CNN models.
    - **examples**: examples and tests of using the cnn models.
  - **utils**: various utilities
- **tensorboard_logs**: path to store tensorboard logs.
- **training_logs**: training logs in CSV to generate learning curves in reports.

### Recipe: Train a model using docker image and cracks dataset

**Note:**  the following instruction steps use a [custom docker image](https://hub.docker.com/r/nhorro/tensorflow1.12-py3-jupyter-opencv) based on official Tensorflow Docker image for GPU. 

Tested with Ubuntu 18.04 and Geforce GTX 950M with nvidia driver version 390.116.

#### Steps

1. Clone repository.

```bash
git clone https://github.com/nhorro/tensorflow-crack-classification.git
```

2. Download crack dataset.

```bash
wget https://data.mendeley.com/datasets/5y9wdsg2zt/1/files/c0d86f9f-852e-4d00-bf45-9a0e24e3b932/Concrete%20Crack%20Images%20for%20Classification.rar
mkdir -pv data/datasets/cracks
unrar x Concrete\ Crack\ Images\ for\ Classification.rar ./data/datasets/cracks
```

3. Split the original dataset in training and evaluation (suggested: 80%-20%):

```bash
python src/utils/split_dataset.py --dir=data/datasets/cracks/ --train=80 --test=20 --output=data/datasets/cracks_splitted8020
```

4. Run docker container for development (note: --network="host" is to enable access to the tensorflow serving API):

```bash
docker run -it --rm --runtime=nvidia -v $(realpath $PWD):/tf/notebooks --name tensorflowdev1 --network="host" nhorro/tensorflow1.12-py3-jupyter-opencv:1.1.0
```

4. Train the model from a notebook:

   1. Open notebook [SimpleNet - Development Notebook](src/cnn_model_development/SimpleNet - Development Notebook.ipynb) and run. 

5. Optional: monitor and debug with tensorboard. 

   1. Install tensorboard if not already installed:

   ```bash
   pip install tensorboard # if not installed
   ```

   Run tensorboard and connect to http://localhost:6006:

   ```bash
   tensorboard --logdir=tensorboard_logs
   ```


### Recipe: Serve a model using tensorflow-serving docker image

Official image tensorflow/serving:1.12.0-gpu is used for serving.

```bash
export SERVING_MODEL=simplenet_cracks8020
docker run -t --rm --runtime=nvidia -p 8501:8501 -v $(realpath $PWD/models):/models/ --name crack_classification_service -e MODEL_NAME=$SERVING_MODEL tensorflow/serving:1.12.0-gpu
```
#### Using the REST API

Query model status to check model availability.

```bash
curl http://localhost:8501/v1/models/simplenet_cracks8020
```

```json
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

Query model metadata to obtain method signature definition (in this case, predict method input is a 64x64x3 image of DT_FLOAT elements).

```bash
 curl http://localhost:8501/v1/models/simplenet_cracks8020/metadata
```

```json
{
"model_spec":{
 "name": "simplenet_cracks8020",
 "signature_name": "",
 "version": "1"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "serving_default": {
   "inputs": {
    "input_image": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "64",
        "name": ""
       },
       {
        "size": "64",
        "name": ""
       },
       {
        "size": "3",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "conv2d_6_input:0"
    }
   },
   "outputs": {
    "dense_7/Softmax:0": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "2",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "dense_7/Softmax:0"
    }
   },
   "method_name": "tensorflow/serving/predict"
  }
 }
}
}
}
```

There is an [example notebook](./src/notebooks/examples/CrackClassificationRESTAPIExample.ipynb) of how to consume the API to classify an scan images.

#### Using the GRPC API

WIP


## References

- [1] Chen, Fu-Chen & Jahanshahi, Mohammad. (2017). NB-CNN: Deep Learning-based Crack Detection Using Convolutional Neural Network and Naïve Bayes Data Fusion. IEEE Transactions on Industrial Electronics. PP. 1-1. 10.1109/TIE.2017.2764844.
- [2] Özgenel, Çağlar Fırat (2018), “Concrete Crack Images for Classification”, Mendeley Data, v1http://dx.doi.org/10.17632/5y9wdsg2zt.1