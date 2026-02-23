# AML Notes - Quick Start Guide

## 🚀 What You Got

A fully functional Next.js application displaying your Applied Machine Learning study notes with:

✅ Dynamic lecture navigation  
✅ Beautiful responsive sidebar  
✅ Markdown content rendering  
✅ Mobile-friendly design  
✅ Production-ready configuration  

## 📁 Project Structure

```
amlnotes/
├── src/
│   ├── app/
│   │   ├── page.tsx              # Home (redirects to lecture-1)
│   │   ├── layout.tsx            # Root layout
│   │   ├── globals.css           # Global styles
│   │   └── [lectureId]/
│   │       └── page.tsx          # Dynamic lecture pages
│   ├── components/
│   │   ├── Sidebar.tsx           # Navigation sidebar
│   │   └── LectureContent.tsx    # Markdown renderer
│   └── data/
│       └── lectures.ts           # All lecture content
├── package.json
├── next.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── Dockerfile
├── docker-compose.yml
├── DEPLOYMENT.md
└── README.md
```

## 🎯 Current Status

- Server running at: **http://localhost:3002** (port 3000 was in use)
- ✅ All 13 lectures loaded
- ✅ Sidebar navigation working
- ✅ Mobile menu toggle functional
- ✅ Markdown rendering active

## 🔧 Common Commands

```bash
# Development
npm run dev          # Start dev server

# Production
npm run build        # Build for production
npm start            # Start production server
npm run lint         # Run ESLint

# Docker
docker build -t amlnotes .          # Build image
docker run -p 3000:3000 amlnotes    # Run container
docker-compose up -d                # Run with compose

# Deployment
vercel                              # Deploy to Vercel
```

## 📝 Editing Content

### Add a New Lecture

Edit `src/data/lectures.ts`:

```typescript
{
  id: "lecture-20",
  number: 20,
  title: "Your New Lecture",
  content: `## Your Content Here
  
This supports markdown formatting...`
}
```

### Customize Colors

**Sidebar:** `src/components/Sidebar.tsx`
- Change `from-blue-900 to-blue-800` to your preferred colors

**Content:** `src/components/LectureContent.tsx`
- Modify Tailwind classes for styling

## 🌐 Deployment (Choose One)

### Option 1: Vercel (Recommended - 2 minutes)
```bash
npm i -g vercel
vercel
```

### Option 2: Docker (Any Server - 10 minutes)
```bash
docker build -t amlnotes .
docker run -p 3000:3000 amlnotes
```

### Option 3: GitHub → Auto-Deploy
- Push to GitHub
- Connect to Vercel/Railway/Render
- Auto-deploys on every push

[See full deployment guide →](./DEPLOYMENT.md)

## 📚 Lectures Included

1. Introduction to Machine Learning
2. Hyperparameters & Tuning
3. Loss Functions (Theory + Implementation)
4. Regression Loss Functions with Numericals
5. Classification Loss Functions
6. Sparse Categorical Loss & Triplet Loss
7. Data Cleaning: Missing Data & Outliers
8. Feature Scaling & Feature Encoding
9. Dimensionality Reduction
10. PCA (Deep Dive)
11. Cross-Validation
12. Handling Imbalanced Data
13. Important Topics for Test

## 🎨 Customization Tips

### Change Sidebar Logo/Text
Edit `src/components/Sidebar.tsx`:
```tsx
<h1 className="text-2xl font-bold">Your App Name</h1>
```

### Add Search Functionality
1. Install: `npm install use-debounce`
2. Add search input in Sidebar
3. Filter lectures by title/content

### Add Dark Mode
1. Install: `npm install next-themes`
2. Wrap app in ThemeProvider
3. Add toggle button in Sidebar

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :3000
kill -9 <PID>
```

### Build Errors
```bash
rm -rf .next node_modules
npm install
npm run build
```

### Docker Issues
```bash
docker system prune           # Clean up
docker build --no-cache .    # Full rebuild
```

## 📊 Performance

- ✅ Static Site Generation (SSG) - faster page loads
- ✅ Code splitting - smaller bundles
- ✅ Lazy loading - optimal memory usage
- ✅ Mobile optimized - responsive design
- ✅ SEO friendly - proper metadata

## 🔐 Security

- ✅ Built-in security headers
- ✅ XSS protection
- ✅ HTTPS auto-enabled (Vercel)
- ✅ Environment variable protection
- ✅ No sensitive data in code

## 📞 Support

- Next.js docs: https://nextjs.org/docs
- Tailwind CSS: https://tailwindcss.com/docs
- React: https://react.dev
- TypeScript: https://www.typescriptlang.org/docs/

## 🎓 Next Steps

1. ✅ Application is running
2. 📦 Customize your content (optional)
3. 🌐 Deploy to production
4. 🔗 Share your app!

---

**Built with Next.js 16.1 + Tailwind CSS 4**  
Created for displaying AML study notes efficiently 📚
