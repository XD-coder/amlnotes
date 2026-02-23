# Deployment Guide for AML Notes

## Quick Deploy Options

### 1. Vercel (Recommended - Free)

**Easiest and fastest way to deploy.**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

Or deploy directly on Vercel.com:
1. Go to https://vercel.com/new
2. Import this GitHub repository
3. Click Deploy
4. Done! Your app is live

### 2. Docker + Local Server

**Deploy on your own server.**

```bash
# Build Docker image
docker build -t amlnotes:latest .

# Run container
docker run -d -p 3000:3000 --name amlnotes amlnotes:latest

# Access at http://localhost:3000
```

Using docker-compose:
```bash
docker-compose up -d
```

### 3. AWS Amplify

1. Push code to GitHub
2. Go to AWS Amplify
3. Connect repository
4. Configure build settings
5. Deploy

### 4. Heroku (Free tier retired, use alternatives)

### 5. Railway.app

1. Connect GitHub repo
2. Railway auto-deploys on push
3. No configuration needed

### 6. Netlify

```bash
npm run build
# Deploy the 'out' directory on Netlify
```

### 7. Google Cloud Run

```bash
# Build and push Docker image
gcloud builds submit --tag gcr.io/PROJECT_ID/amlnotes

# Deploy
gcloud run deploy amlnotes --image gcr.io/PROJECT_ID/amlnotes
```

### 8. Azure Static Web Apps

1. Connect GitHub repository
2. Azure auto-configures for Next.js
3. Deploy on push

## Environment Variables

Create `.env.local` file:
```
NEXT_PUBLIC_APP_NAME=AML Notes
NODE_ENV=production
```

## Performance Tips

- Vercel handles Next.js optimization automatically
- Docker builds are optimized for production
- All deployments use static site generation (SSG)
- Automatic code splitting reduces bundle size

## Monitoring & Logs

**Vercel:**
- Check analytics at vercel.com dashboard
- View logs in real-time

**Docker:**
```bash
docker logs -f amlnotes
```

**Railway/Render:**
- View logs in dashboard

## Custom Domain

**Vercel:**
1. Add domain in Settings
2. Update DNS records
3. Auto HTTPS enabled

**Others:**
Update DNS CNAME records to point to your deployed URL

## Troubleshooting

### Port Already in Use
```bash
# Find process
lsof -i :3000

# Kill it
kill -9 <PID>
```

### Build Fails
```bash
# Clear cache
rm -rf .next
npm install
npm run build
```

### Docker Issues
```bash
# Rebuild without cache
docker build --no-cache -t amlnotes .

# Clean up
docker system prune
```

## Cost Comparison

| Platform | Cost | Setup Time |
|----------|------|-----------|
| Vercel | Free (up to 100 GB/month) | 2 minutes |
| Railway | $5/month | 5 minutes |
| AWS Amplify | Free tier (15GB/month) | 10 minutes |
| Google Cloud Run | Pay as you go | 15 minutes |
| Azure | Free tier available | 20 minutes |
| DigitalOcean | $5/month | 20 minutes |

## Scaling Considerations

All options scale automatically except:
- Self-hosted Docker (manual scaling needed)
- Needs to handle spikes during test season

## Security

- All deployments use HTTPS by default
- Environment variables are encrypted
- No sensitive data in code
- Next.js handles security headers

---

**Recommended for beginners:** Vercel (one-click deploy)
**Recommended for full control:** Docker on DigitalOcean or AWS
**Recommended for academic use:** Railway or Render
