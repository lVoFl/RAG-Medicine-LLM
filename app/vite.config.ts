import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
  ],
  server: {
    host: '0.0.0.0',
    port: 8080,
    allowedHosts: ['i-2.gpushare.com'],
  },
  preview: {
    host: '0.0.0.0',
    port: 8080,
    allowedHosts: ['i-2.gpushare.com'],
  },
})
