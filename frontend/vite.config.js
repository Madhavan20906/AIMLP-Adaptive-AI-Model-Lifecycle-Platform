import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/app/',
  plugins: [react()],
  server: {
    proxy: {
      '/mode1': 'http://localhost:8000',
      '/mode2': 'http://localhost:8000',
      '/registry': 'http://localhost:8000',
    },
  },
})
