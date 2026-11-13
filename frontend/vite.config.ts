import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发期代理到 FastAPI（同源免 CORS）；生产由 FastAPI 托管 dist。
export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      output: {
        // 依赖分包：业务代码迭代时 vendor 缓存不变
        manualChunks: {
          vue: ['vue', 'vue-router', 'pinia'],
          'element-plus': ['element-plus'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
