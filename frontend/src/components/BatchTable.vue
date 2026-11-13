<script setup lang="ts">
import type { BatchItem } from '../api/types'

defineProps<{ items: BatchItem[]; running: Record<number, boolean> }>()

const STATUS_LABEL: Record<string, string> = {
  pending: '等待中',
  ok: '成功',
  failed: '失败',
  cancelled: '已取消',
  skipped: '跳过',
}

function statusType(status: string, isRunning: boolean) {
  if (isRunning) return 'primary'
  if (status === 'ok') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'cancelled') return 'warning'
  return 'info'
}
</script>

<template>
  <el-table :data="items" size="small" stripe>
    <el-table-column prop="item_index" label="#" width="60" />
    <el-table-column prop="image_name" label="文件名" min-width="160" show-overflow-tooltip />
    <el-table-column label="状态" width="110">
      <template #default="{ row }">
        <el-tag :type="statusType(row.status, !!running[row.item_index])" size="small">
          {{ running[row.item_index] ? '修复中' : (STATUS_LABEL[row.status] ?? row.status) }}
        </el-tag>
      </template>
    </el-table-column>
    <el-table-column label="耗时" width="100">
      <template #default="{ row }">
        {{ row.latency_ms != null ? `${row.latency_ms.toFixed(0)} ms` : '—' }}
      </template>
    </el-table-column>
    <el-table-column label="掩码比例" width="100">
      <template #default="{ row }">
        {{ row.mask_ratio != null ? `${(row.mask_ratio * 100).toFixed(1)}%` : '—' }}
      </template>
    </el-table-column>
    <el-table-column prop="error_code" label="错误码" width="140" />
    <el-table-column label="结果" width="90">
      <template #default="{ row }">
        <el-link v-if="row.output_url" type="primary" :href="row.output_url" target="_blank">
          查看
        </el-link>
        <span v-else>—</span>
      </template>
    </el-table-column>
  </el-table>
</template>
