#!/usr/bin/env bash

set -e

echo "=== Installing ArgoCD ==="

# 1. Create namespace for ArgoCD
kubectl create namespace argocd || true

# 2. Apply ArgoCD CRDs and components
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# echo "Waiting for ArgoCD pods to be ready..."
# kubectl wait --for=condition=Ready pods -n argocd --all --timeout=180s

# 2. Expose ArgoCD server via NodePort
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "NodePort"}}'

# 3. Create namespace for ArgoCD Destination
kubectl create namespace mlops-dev || true

# 4. Get NodePort and initial password
NODE_PORT=$(kubectl get svc argocd-server -n argocd -o jsonpath='{.spec.ports[0].nodePort}')
ADMIN_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

echo ""
echo "ArgoCD server is now accessible on:"
echo "  http://<control-plane-ip>:$NODE_PORT"
echo ""
echo "Initial login credentials:"
echo "  username: admin"
echo "  password: $ADMIN_PASSWORD"
echo ""
echo "=== ArgoCD setup complete ==="
