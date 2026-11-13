<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const active = computed(() => String(route.name ?? 'inpaint'))
</script>

<template>
  <el-container class="layout">
    <el-aside width="200px" class="aside">
      <div class="brand">
        <span class="brand-title">Hybrid Inpaint</span>
        <span class="brand-sub">U-Net + Transformer 图像修复</span>
      </div>
      <el-menu :default-active="active" router class="menu">
        <el-menu-item index="/inpaint" route="/inpaint"> 单图修复 </el-menu-item>
        <el-menu-item index="/batch" route="/batch"> 批量修复 </el-menu-item>
        <el-menu-item index="/history" route="/history"> 历史记录 </el-menu-item>
      </el-menu>
    </el-aside>
    <el-main class="main">
      <router-view />
    </el-main>
  </el-container>
</template>

<style scoped>
.layout {
  min-height: 100vh;
}
.aside {
  background: #f5f7fa;
  border-right: 1px solid #e4e7ed;
}
.brand {
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.brand-title {
  font-size: 18px;
  font-weight: 700;
}
.brand-sub {
  font-size: 12px;
  color: #909399;
}
.menu {
  border-right: none;
  background: transparent;
}
.main {
  padding: 24px;
}
</style>
