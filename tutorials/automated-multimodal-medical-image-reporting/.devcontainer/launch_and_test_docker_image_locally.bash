docker run --name ammir_container --detach ammir:latest # -v $HOME/datasets/chest-xrays-indiana-university:datasets
docker exec -it ammir_container bash

# docker run -d -v <local path>:<container-path> <docker-image-name>
# docker exec -it <> bash <docker-container-id> bash #or <NAME> bash
