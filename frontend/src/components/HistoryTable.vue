<script setup lang="ts">
import type { HistoryItem } from '../api/types'

defineProps<{ items: HistoryItem[]; loading: boolean }>()
const emit = defineEmits<{ remove: [jobId: string, deleteFiles: boolean] }>()

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString('zh-CN', { hour12: false })
}
</script>

<template>
  <el-table v-loading="loading" :data="items" size="small" stripe>
    <el-table-column label="结果" width="90">
      <template #default="{ row }">
        <el-image
          :src="row.output_url"
          fit="cover"
          class="thumb"
          :preview-src-list="[row.output_url]"
          preview-teleported
        />
      </template>
    </el-table-column>
    <el-table-column label="时间" width="170">
      <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
    </el-table-column>
    <el-table-column prop="mode" label="来源" width="70" />
    <el-table-column label="尺寸" width="100">
      <template #default="{ row }">{{ row.width }}×{{ row.height }}</template>
    </el-table-column>
    <el-table-column label="掩码比例" width="90">
      <template #default="{ row }">{{ (row.mask_ratio * 100).toFixed(1) }}%</template>
    </el-table-column>
    <el-table-column label="耗时" width="90">
      <template #default="{ row }">{{ row.latency_ms.toFixed(0) }} ms</template>
    </el-table-column>
    <el-table-column prop="note" label="备注" min-width="120" show-overflow-tooltip />
    <el-table-column label="操作" width="150">
      <template #default="{ row }">
        <el-link type="primary" :href="row.output_url" download class="op">下载</el-link>
        <el-popconfirm
          title="删除该记录？"
          confirm-button-text="删除"
          cancel-button-text="取消"
          @confirm="emit('remove', row.job_id, false)"
        >
          <template #reference>
            <el-link type="danger" class="op">删除</el-link>
          </template>
        </el-popconfirm>
        <el-popconfirm
          title="同时删除图片文件？"
          confirm-button-text="全部删除"
          cancel-button-text="取消"
          @confirm="emit('remove', row.job_id, true)"
        >
          <template #reference>
            <el-link type="danger" class="op">删文件</el-link>
          </template>
        </el-popconfirm>
      </template>
    </el-table-column>
  </el-table>
</template>

<style scoped>
.thumb {
  width: 56px;
  height: 56px;
  border-radius: 4px;
}
.op {
  margin-right: 8px;
}
</style>
