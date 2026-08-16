<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '../stores/app'
import ChatHeader from '../components/ChatHeader.vue'
import MessageList from '../components/MessageList.vue'
import Composer from '../components/Composer.vue'

const store = useAppStore()
const hasWorkspace = computed(() => !!store.person && !!store.conversation)
</script>

<template>
  <section class="empty-chat" v-if="!hasWorkspace">
    <div class="empty-inner">
      <p class="kicker">PCFM · 对话式人物模拟</p>
      <h2>选择或新建一个人物</h2>
      <p>从真实资料出发，进行有证据边界的多轮交流。</p>
      <button class="button primary" @click="store.openCreatePerson()">创建第一个人物</button>
    </div>
  </section>
  <section class="chat-workspace" v-else>
    <ChatHeader />
    <MessageList />
    <Composer />
  </section>
</template>
