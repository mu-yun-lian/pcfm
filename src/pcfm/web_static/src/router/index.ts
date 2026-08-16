import { createRouter, createWebHashHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'

// Hash history keeps deep links working when the backend serves the built app
// from /dist/ without an SPA-aware fallback for arbitrary paths.
const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', name: 'chat', component: ChatView },
  ],
})

export default router
