#!/usr/bin/env bash

# Import local images into containerd
ctr -n k8s.io images import /vagrant/build/ml-train.tar
ctr -n k8s.io images import /vagrant/build/ml-api.tar