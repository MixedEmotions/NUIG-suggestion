PYVERSION=3.5
NAME=16_suggestion_mining_nuig
REPO=mixedemotions
VERSION=0.3.2
PLUGINS= $(filter %/, $(wildcard */))


all: build run

build: clean Dockerfile
	docker build -t '$(REPO)/$(NAME):$(VERSION)-python$(PYVERSION)' -f Dockerfile .;

test-%:
	docker run -v $$PWD/$*:/senpy-plugins/ --rm --entrypoint=/usr/local/bin/py.test -ti '$(REPO)/$(NAME):$(VERSION)-python$(PYVERSION)' test.py

test: $(addprefix test-,$(PLUGINS))

clean:
	@docker ps -a | awk '/$(REPO)\/$(NAME)/{ split($$2, vers, "-"); if(vers[1] != "${VERSION}"){ print $$1;}}' | xargs docker rm 2>/dev/null|| true
	@docker images | awk '/$(REPO)\/$(NAME)/{ split($$2, vers, "-"); if(vers[1] != "${VERSION}"){ print $$1":"$$2;}}' | xargs docker rmi 2>/dev/null|| true

run: build
	docker run --rm -p 5001:5000 -ti '$(REPO)/$(NAME):$(VERSION)-python$(PYVERSION)'

.PHONY: test test-% build-% build test test_pip run clean
