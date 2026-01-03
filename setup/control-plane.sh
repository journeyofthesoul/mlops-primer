#!/usr/bin/env bash

if [ -z ${K8S_VERSION+x} ]; then
  K8S_VERSION=1.31.1-1.1
fi

POD_CIDR=$1
API_ADV_ADDRESS=$2

# Initialize control plane
mkdir -p /vagrant/infra-context
kubeadm init --kubernetes-version v${K8S_VERSION:0:6} --pod-network-cidr $POD_CIDR --apiserver-advertise-address $API_ADV_ADDRESS | tee /vagrant/infra-context/kubeadm-init.out

# Configure kubelet
systemctl daemon-reload
echo "KUBELET_EXTRA_ARGS=--node-ip=$API_ADV_ADDRESS --cgroup-driver=systemd" > /etc/default/kubelet
systemctl restart kubelet

# Configure kubectl for vagrant and root users
mkdir -p /home/vagrant/.kube
cp -i /etc/kubernetes/admin.conf /home/vagrant/.kube/config
chown vagrant:vagrant /home/vagrant/.kube/config
mkdir -p /root/.kube
cp -i /etc/kubernetes/admin.conf /root/.kube/config

# Install Calico networking
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/tigera-operator.yaml
wget https://raw.githubusercontent.com/projectcalico/calico/v3.29.1/manifests/custom-resources.yaml
sed -i 's~cidr: 192\.168\.0\.0/16~cidr: 172\.18\.0\.0/16~g' custom-resources.yaml
kubectl create -f custom-resources.yaml
rm custom-resources.yaml

# Copy admin.conf to /vagrant for host use
chown vagrant:vagrant /vagrant/infra-context
cp /etc/kubernetes/admin.conf /vagrant/infra-context/admin.conf
chown vagrant:vagrant /vagrant/infra-context/admin.conf

# Rewrite API server for host kubectl
cp /vagrant/infra-context/admin.conf /vagrant/infra-context/local-admin.conf
if command -v yq >/dev/null 2>&1; then
  yq eval '.clusters[0].cluster.server = "https://127.0.0.1:6443"' -i /vagrant/infra-context/local-admin.conf
else
  sed -i.bak 's~server: https://.*:6443~server: https://127.0.0.1:6443~' /vagrant/infra-context/local-admin.conf
fi