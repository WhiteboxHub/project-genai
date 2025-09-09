#!/bin/bash

# Create required directories for Milvus
mkdir -p volumes/etcd volumes/minio volumes/milvus

# Pull required Docker images
docker pull quay.io/coreos/etcd:v3.5.18
docker pull minio/minio:RELEASE.2024-12-18T13-15-44Z
docker pull milvusdb/milvus:v2.6.1

# Start Milvus services
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 30

# Check service health
echo "Checking service health..."
curl -f http://localhost:9091/healthz || echo "Milvus is not healthy"
curl -f http://localhost:9000/minio/health/live || echo "MinIO is not healthy"

echo "Setup complete! Milvus should be accessible at:"
echo "- Milvus API: http://localhost:19530"
echo "- Milvus admin: http://localhost:9091"
echo "- MinIO console: http://localhost:9001"