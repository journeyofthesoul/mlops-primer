#!/usr/bin/env bash

set -e

mkdir -p ./build

echo "Building images on host"
docker build -t localhost/ml-train:latest ./train
docker build -t localhost/ml-api:latest ./api

echo "Packaging images to tar"
docker save localhost/ml-train:latest -o ./build/ml-train.tar
docker save localhost/ml-api:latest -o ./build/ml-api.tar