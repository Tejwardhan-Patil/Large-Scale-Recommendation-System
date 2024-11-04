#!/bin/bash

# Ensure script is being run with superuser privileges
if [ "$(id -u)" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

# Set environment variables
export REPO_DIR="/repo"
export DOCKER_IMAGE="recommender-system:latest"
export KUBERNETES_DIR="$REPO_DIR/model-serving/Kubernetes"
export HELM_CHART_DIR="$REPO_DIR/helm"

# Step 1: Build Docker Images
echo "Building Docker images..."
docker build -t $DOCKER_IMAGE -f $REPO_DIR/Dockerfile_recommender $REPO_DIR

# Step 2: Push Docker Images
echo "Pushing Docker images to registry..."
docker tag $DOCKER_IMAGE registry/$DOCKER_IMAGE
docker push registry/$DOCKER_IMAGE

# Step 3: Apply Kubernetes Manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f $KUBERNETES_DIR/deployment.yaml
kubectl apply -f $KUBERNETES_DIR/service.yaml

# Step 4: Deploy with Helm
echo "Deploying Helm charts..."
helm install recommender-system $HELM_CHART_DIR/recommender_helm_chart.yaml

# Step 5: Check Deployment Status
echo "Checking deployment status..."
kubectl get pods
kubectl get services

echo "Deployment complete!"