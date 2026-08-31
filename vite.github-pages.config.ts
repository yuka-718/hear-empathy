import tailwindcss from '@tailwindcss/postcss';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const projectRoot = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  root: 'github-pages',
  base: '/hear-empathy/',
  publicDir: fileURLToPath(new URL('public', import.meta.url)),
  resolve: {
    alias: {
      '@': projectRoot,
    },
  },
  css: {
    postcss: {
      plugins: [tailwindcss()],
    },
  },
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL('docs', import.meta.url)),
    emptyOutDir: true,
  },
});
