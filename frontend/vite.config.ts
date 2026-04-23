import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:4646',
      '/ws': {
        target: 'http://localhost:4646',
        ws: true
      },
      '/qlms': 'http://localhost:4646'
    }
  },
  build: {
    modulePreload: false,
    outDir: 'dist'
  }
})
