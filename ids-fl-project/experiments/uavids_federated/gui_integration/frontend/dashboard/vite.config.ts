import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Port 3000 matches the backend's default allowed CORS origin
// (see gui_integration/backend.py DEFAULT_ORIGINS / start_backend.ps1 -FrontendOrigin).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    strictPort: true,
  },
})
