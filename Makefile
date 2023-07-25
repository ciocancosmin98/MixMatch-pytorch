.PHONY: pimage pclean prun pcleanall

TAG=ubuntu:mixmatch
DOCKERFILE=./podman/CUDA11.Dockerfile
CONTAINERNAME=mixmatch2

pimage:
	sudo podman build --tag $(TAG) -f $(DOCKERFILE)
pclean:
	sudo podman rmi $(TAG)
prun:
	sudo podman run -d -it --privileged --name=$(CONTAINERNAME) -p 6006:6006 --shm-size=4g -v .:/mixmatch --security-opt=label=disable $(TAG)
pcleanall:
	sudo podman system prune --all --force && podman rmi --all
