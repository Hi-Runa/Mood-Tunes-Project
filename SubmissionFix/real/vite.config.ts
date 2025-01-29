/*
        Ayush Vupalanchi, Vaibhav Alaparthi, Hiruna Devadithya
        1/23/25

        Vite bundler configuration with React plugin and optimization settings.

/*

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
});
