<script setup lang="ts">
import { onMounted } from 'vue'
import HistoryTable from '../components/HistoryTable.vue'
import { exportUrl } from '../api/history'
import { useHistoryStore } from '../stores/history'

const store = useHistoryStore()

onMounted(() => store.load())

function onRemove(jobId: string, deleteFiles: boolean) {
  void store.remove(jobId, deleteFiles)
}

function onPage(page: number) {
  void store.setPage((page - 1) * store.limit)
}
</script>

<template>
  <div class="history-view">
    <div class="header">
      <h2>历史记录</h2>
      <div>
        <el-button size="small" @click="store.load()">刷新</el-button>
        <el-link type="primary" :href="exportUrl()" download class="export">导出 CSV</el-link>
      </div>
    </div>
    <el-alert v-if="store.error" :title="store.error" type="error" show-icon class="alert" />
    <HistoryTable :items="store.items" :loading="store.loading" @remove="onRemove" />
    <el-pagination
      v-if="store.items.length || store.offset > 0"
      layout="prev, pager, next"
      :page-size="store.limit"
      :current-page="Math.floor(store.offset / store.limit) + 1"
      :total="store.offset + store.items.length + (store.hasMore ? store.limit : 0)"
      class="pager"
      @current-change="onPage"
    />
  </div>
</template>

<style scoped>
.history-view {
  max-width: 1080px;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.export {
  margin-left: 12px;
}
.alert {
  margin: 12px 0;
}
.pager {
  margin-top: 16px;
  justify-content: flex-end;
}
</style>
