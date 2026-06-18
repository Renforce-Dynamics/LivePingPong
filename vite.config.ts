import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: {
    headers: {
      // Required if we later switch to the multi-threaded WASM build.
      // 'Cross-Origin-Opener-Policy': 'same-origin',
      // 'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
