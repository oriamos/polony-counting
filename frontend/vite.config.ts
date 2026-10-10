import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Dev: Vite proxies /api to the FastAPI backend (default 127.0.0.1:8000).
// Use API_TARGET=http://127.0.0.1:8787 npm run dev to develop against the mock.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.API_TARGET || 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
