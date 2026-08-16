import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// The Python backend serves static files from web_static (STATIC_DIR) and keeps
// index.html as the SPA fallback. We build to web_static/dist and use relative
// asset paths so dist/index.html resolves ./assets/* against /dist/.
export default defineConfig({
  base: './',
  plugins: [vue()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
  },
})
