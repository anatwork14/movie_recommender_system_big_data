import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiHost = globalThis.process?.env?.VITE_API_HOST || 'localhost'
const apiPort = globalThis.process?.env?.VITE_API_PORT || 5001

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': `http://${apiHost}:${apiPort}`,
    },
    host: '0.0.0.0',
    port: 5173,
  },
})
