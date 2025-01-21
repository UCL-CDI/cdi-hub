
## Build
```
docker compose -f docker-compose.yml build #Building estimated time

# docker images
#REPOSITORY   TAG               IMAGE ID       CREATED         SIZE
#awsbatch     latest            811d5b0b07f9   2 minutes ago   452MB
```

## Launch and test image
```
docker run --name awsbatch_container --detach awsbatch:latest
docker exec -it awsbatch_container bash
```

## Stop container and remove it
```
bash ../stop_container_and_removeit.bash
```

