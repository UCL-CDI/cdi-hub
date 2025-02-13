# Run container in detached mode
docker run --name awsbatch_container -v $HOME/datasets/chest-xrays-indiana-university:/datasets/chest-xrays-indiana-university --gpus all --detach awsbatch:latest

# Access container shell for testing
docker exec -it awsbatch_container bash
