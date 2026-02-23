# AML Notes - Applied Machine Learning Study App

A beautiful, responsive Next.js application for displaying comprehensive Applied Machine Learning study notes with sidebar navigation.

## Features

✨ **Modern Design**
- Clean, professional UI with Tailwind CSS  
- Fully responsive (mobile, tablet, desktop)
- Smooth navigation and transitions

📚 **Complete Content**
- 13 lectures on Machine Learning fundamentals
- Topics: Loss Functions, Regression, Classification, Feature Engineering, PCA, Cross-Validation, Data Cleaning

🗂️ **Organized Navigation**
- Sidebar with quick access to all lectures
- Mobile-friendly menu toggle
- Current lecture highlighting

## Quick Start

```bash
# Install dependencies
npm install

# Start development server  
npm run dev

# Open http://localhost:3000 in browser
```

## Build & Deploy

```bash
npm run build
npm start
```

### Deploy to Vercel (Recommended)
1. Push code to GitHub
2. Import repository at vercel.com
3. Deploy automatically

### Docker
```bash
docker build -t amlnotes .
docker run -p 3000:3000 amlnotes
```

## Technologies
- Next.js 16.1 + TypeScript
- Tailwind CSS 4
- react-markdown with remark-gfm

**Created by:** Kartikey Mishra  
**Happy Learning!** 📚
