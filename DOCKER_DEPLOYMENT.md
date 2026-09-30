# Docker Deployment Guide

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Copy and configure environment file
cp .env.docker .env.docker.local

# Edit with your credentials (email, database, JWT secret)
nano .env.docker.local  # or use your editor

# Build and start all services
docker-compose up --build
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 2. Services Running

| Service | Port | Container | Status |
|---------|------|-----------|--------|
| Frontend (React) | 3000 | vtds-frontend | Health check enabled |
| Backend (FastAPI) | 8000 | vtds-backend | Health check enabled |
| Database (PostgreSQL) | 5432 | vtds-db | Health check enabled |

---

## Configuration

### Using Environment Variables

Create `.env.docker.local` with your settings:

```bash
# Database
DB_USER=your_db_user
DB_PASSWORD=your_secure_password
DB_NAME=vehicle_theft_db

# Email (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=your-email@gmail.com
EMAIL_RECIPIENTS=recipient@example.com

# Authentication
JWT_SECRET_KEY=generate-a-random-long-string-here

# API Base URL (keep as is for Docker internal communication)
VITE_API_BASE_URL=http://backend:8000
```

---

## Common Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f db
```

### Rebuild After Code Changes
```bash
docker-compose up --build
```

### Access Database Shell
```bash
docker-compose exec db psql -U vtds_user -d vehicle_theft_db
```

### Initialize Database with Seed Data
```bash
docker-compose exec backend python -m backend.database.seed --demo
```

---

## Production Deployment

### For AWS, Azure, GCP, etc.:

1. **Push to Container Registry:**
   ```bash
   # AWS ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
   docker build -f Dockerfile.backend -t 123456789.dkr.ecr.us-east-1.amazonaws.com/vtds-backend:latest .
   docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/vtds-backend:latest
   ```

2. **Environment Configuration:**
   - Use a managed database (RDS, CloudSQL, Azure Database)
   - Store secrets in (AWS Secrets Manager, Azure Key Vault, Google Secret Manager)
   - Use external SMTP or Resend API for email
   - Set strong JWT secret
   - Update API URLs to production domain

3. **Deploy with Orchestration:**
   - Kubernetes: Use helm charts or kubectl manifests
   - Docker Swarm: Use docker-compose in swarm mode
   - AWS ECS: Use task definitions
   - DigitalOcean App Platform: Use docker-compose.yml directly

### Database Migration (Production)
```bash
docker-compose exec backend alembic upgrade head
```

---

## Troubleshooting

### Backend won't start: "Connection refused"
- Database may not be healthy yet
- Wait 30-40 seconds and check logs: `docker-compose logs backend`
- Verify DATABASE_URL in your .env file

### Frontend shows "Cannot reach backend"
- Check VITE_API_BASE_URL is set correctly
- For external access, update to your domain: `https://api.yourdomain.com`
- In docker-compose, use `http://backend:8000` for internal communication

### Models not found
- Ensure `models/` directory exists with all .pt and .onnx files
- Volume mapping in docker-compose.yml includes `./models:/app/models`

### Email not sending
- Verify SMTP credentials in .env file
- Check logs: `docker-compose logs backend | grep -i email`
- For Gmail, use **App Password** not regular password
- If using Resend, set `EMAIL_PROVIDER=resend` and verify API key

### Database grows too large
- Clean up old alerts/evidence periodically
- Consider archiving to S3/blob storage
- Monitor disk usage: `docker exec vtds-db du -sh /var/lib/postgresql/data`

---

## Performance Tips

- **MLModels**: Pre-download models to `./models/` before building (faster startup)
- **Database**: Tune PostgreSQL config for your hardware
- **Backend Workers**: Increase Uvicorn workers: `uvicorn backend.main:app --workers 4`
- **Caching**: Add Redis for session/cache layer
- **Storage**: Use S3/Blob storage for evidence videos instead of local `./data`

---

## Volume Mounts

| Path | Purpose | Persistent? |
|------|---------|-------------|
| `./data` | User uploads, evidence, faces | Yes (important) |
| `./models` | ML model files (.pt, .onnx) | Yes (read-only) |
| `./logs` | Application logs | Yes |
| `postgres_data` | Database files | Yes (Docker managed) |

---

## Next Steps

1. Configure email provider (Gmail app password or Resend API key)
2. Set strong JWT_SECRET_KEY
3. Test: `curl http://localhost:8000/health`
4. Deploy to production with managed database & container registry
5. Monitor logs and set up alerting
