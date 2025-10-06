"""
Milvus Setup Instructions

1. Prerequisites:
- Docker and Docker Compose installed
- At least 2GB of free memory
- Port 19530 and 9091 available

2. Directory Structure:
Create the following directory structure for persistent storage:
./volumes/
  ├── etcd/       # For metadata storage
  ├── minio/      # For object storage
  └── milvus/     # For Milvus data

3. Environment Variables:
DOCKER_VOLUME_DIRECTORY=./volumes  # Optional, defaults to ./volumes

4. Start Milvus:
cd docker-files
docker-compose up -d

5. Verify Installation:
- Milvus API: http://localhost:19530
- Milvus admin: http://localhost:9091
- MinIO console: http://localhost:9001 (credentials: minioadmin/minioadmin)

6. Stop Milvus:
docker-compose down
"""