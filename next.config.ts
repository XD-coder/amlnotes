import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // Enable React strict mode for development warnings
  reactStrictMode: true,
  
  // Configure image optimization
  images: {
    unoptimized: true, // Useful for static exports if needed
  },
  
  // Add security headers
  headers: async () => {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "X-XSS-Protection",
            value: "1; mode=block",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
