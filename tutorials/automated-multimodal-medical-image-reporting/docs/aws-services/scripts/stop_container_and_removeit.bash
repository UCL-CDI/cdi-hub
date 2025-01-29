# Clean up after testing
docker stop $(docker ps -a -q)
docker system prune -f --volumes #clean unused systems

#?docker stop batch_demo
#?docker rm batch_demo
