<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  done: number
  failed: number
  total: number
  status: string
}>()

const percent = computed(() =>
  props.total === 0 ? 0 : Math.round(((props.done + props.failed) / props.total) * 100),
)

const STATUS_LABEL: Record<string, string> = {
  queued: '排队中',
  running: '进行中',
  succeeded: '全部成功',
  failed: '存在失败',
  cancelled: '已取消',
}
</script>

<template>
  <div class="progress">
    <el-progress
      :percentage="percent"
      :status="status === 'succeeded' ? 'success' : status === 'failed' ? 'exception' : undefined"
    />
    <span class="label">
      {{ STATUS_LABEL[status] ?? status }} · 成功 {{ done }} / 失败 {{ failed }} / 共 {{ total }}
    </span>
  </div>
</template>

<style scoped>
.progress {
  display: flex;
  align-items: center;
  gap: 16px;
}
.progress :deep(.el-progress) {
  flex: 1;
}
.label {
  font-size: 13px;
  color: #606266;
  white-space: nowrap;
}
</style>
