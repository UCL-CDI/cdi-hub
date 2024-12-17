# Docker

## Build
```
docker compose -f .devcontainer/docker-compose.yml build

docker images
#REPOSITORY                TAG               IMAGE ID       CREATED          SIZE
#template-project2-image   latest            570651c23f32   45 seconds ago   7.86GB
```

## Commands
```
docker images
docker ps
docker attach <ID>
docker stop <ID>
docker rename keen_einstein mycontainer
docker rmi --force <ID>
```

## References
https://dev.to/behainguyen/python-docker-image-build-install-required-packages-via-requirementstxt-vs-editable-install-572j 

