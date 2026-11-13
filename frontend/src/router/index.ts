import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/inpaint' },
    {
      path: '/inpaint',
      name: 'inpaint',
      component: () => import('../views/InpaintView.vue'),
      meta: { title: '单图修复' },
    },
    {
      path: '/batch',
      name: 'batch',
      component: () => import('../views/BatchView.vue'),
      meta: { title: '批量修复' },
    },
    {
      path: '/history',
      name: 'history',
      component: () => import('../views/HistoryView.vue'),
      meta: { title: '历史记录' },
    },
  ],
})

router.afterEach((to) => {
  document.title = `${String(to.meta.title ?? '')} · Hybrid Inpaint`
})

export default router
