import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = globalThis.process?.env?.VITE_API_PORT || 5000

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': `http://localhost:${apiPort}`,
    },
    host: '0.0.0.0',
    port: 5173,
  },
})
