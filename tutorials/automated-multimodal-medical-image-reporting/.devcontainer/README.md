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
docker image prune -a #clean unused images
```

## References
https://dev.to/behainguyen/python-docker-image-build-install-required-packages-via-requirementstxt-vs-editable-install-572j 

## Solved issues
* `pass not initialized: exit status 1`
```
Error saving credentials: error storing credentials - err: exit status 1, out: `error storing credentials - err: exit status 1, out: `pass not initialized: exit status 1: Error: password store is empty. Try "pass init".``
```
SOLVED:  "Remove the key credStore from ~/.docker/config.json" > https://stackoverflow.com/questions/71770693
