docker run -it --rm --runtime=nvidia -v $(realpath $PWD):/tf/notebooks --name tensorflowdev1 --network="host" nhorro/tensorflow1.12-py3-jupyter-opencv:1.1.0
