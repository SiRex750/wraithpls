import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: false,
    // Allow serving files from one level up to let us fetch from http://localhost:8000 if needed
    fs: { allow: ['..'] }
  }
})
