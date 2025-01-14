## Docker and data management

### Build
```
docker compose -f docker-compose.yml build #Building estimated time

#$docker images
#REPOSITORY   TAG     IMAGE ID      CREATED         SIZE
#ammir        latest  <ID>          current_time    16.8GB
```

### Launch and test image
```
bash launch_and_test_docker_image_locally.bash
```


### Commands
```
docker images
docker ps
docker attach <ID>
docker stop <ID>
docker stop $(docker ps -a -q) # stop all containers
docker rename keen_einstein mycontainer
docker rmi --force <ID>
docker image prune -a #clean unused images
docker system prune -f --volumes #clean unused systems
docker inspect <container-name> (or <container-id>) 
docker volume ls
docker volume rm  <ID>
```

### References
* [python-docker-image-build-install-required-packages](https://dev.to/behainguyen/python-docker-image-build-install-required-packages-via-requirementstxt-vs-editable-install-572j)
* [speed-up-your-docker-builds-with-cache-from](https://lipanski.com/posts/speed-up-your-docker-builds-with-cache-from)
* [docker-cheatsheets](https://github.com/cheat/cheatsheets/blob/master/docker)
