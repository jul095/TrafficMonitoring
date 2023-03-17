# Variables
IMAGENAME=traffic-monitoring
VERSION=v1
AZ_REGISTRY=fkkstudents

# Generic variables
REGISTRY=${AZ_REGISTRY}.azurecr.io
IMAGEFULLNAME=${REGISTRY}/${IMAGENAME}:${VERSION}

.PHONY: help build push all

help:
	    @echo "Makefile commands:"
	    @echo ""
	    @echo "build"
	    @echo "push"
	    @echo "all"

.DEFAULT_GOAL := all

build:
	    @docker build --pull -t ${IMAGEFULLNAME} .

push: build login
	    @docker push ${IMAGEFULLNAME}

login:
		@az acr login --name ${AZ_REGISTRY}

all: build login push
