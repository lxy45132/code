<script setup lang="ts">
import { computed } from 'vue'
import { ElMessage, type UploadFile, type UploadRawFile } from 'element-plus'
import BatchTable from '../components/BatchTable.vue'
import ProgressBar from '../components/ProgressBar.vue'
import { reportUrl } from '../api/batches'
import { useBatchStore } from '../stores/batch'

const store = useBatchStore()

const canSubmit = computed(
  () => store.images.length > 0 && store.images.length === store.masks.length,
)

function onImagesChange(_file: UploadFile, fileList: UploadFile[]) {
  store.setImages(fileList.map((f) => f.raw).filter((f): f is UploadRawFile => !!f))
}

function onMasksChange(_file: UploadFile, fileList: UploadFile[]) {
  store.setMasks(fileList.map((f) => f.raw).filter((f): f is UploadRawFile => !!f))
}

async function submit() {
  await store.submit()
  if (store.error) ElMessage.error(store.error)
}

async function cancel() {
  await store.cancel()
  ElMessage.info('已请求取消，worker 完成当前项后停止')
}
</script>

<template>
  <div class="batch-view">
    <h2>批量修复</h2>
    <div class="uploads">
      <el-upload
        multiple
        :auto-upload="false"
        accept="image/*"
        :on-change="onImagesChange"
        :disabled="store.running"
        drag
        class="upload-box"
      >
        <div class="tip">图片（可多选，≤32 张）</div>
      </el-upload>
      <el-upload
        multiple
        :auto-upload="false"
        accept="image/*"
        :on-change="onMasksChange"
        :disabled="store.running"
        drag
        class="upload-box"
      >
        <div class="tip">掩码（与图片按顺序一一对应，白=洞）</div>
      </el-upload>
    </div>
    <p class="pair-hint">
      已选图片 {{ store.images.length }} 张 / 掩码 {{ store.masks.length }} 张
      <span v-if="!canSubmit && store.images.length">（数量需一致）</span>
    </p>

    <div class="actions">
      <el-input v-model="store.note" placeholder="批次备注（可选）" class="note" maxlength="200" />
      <el-button
        type="primary"
        :disabled="!canSubmit"
        :loading="store.submitting"
        @click="submit"
      >
        提交批量
      </el-button>
      <el-button v-if="store.running" type="warning" @click="cancel">取消</el-button>
      <el-button @click="store.reset()">重置</el-button>
    </div>

    <el-alert v-if="store.error" :title="store.error" type="error" show-icon class="alert" />

    <template v-if="store.detail">
      <ProgressBar
        :done="store.detail.done"
        :failed="store.detail.failed"
        :total="store.detail.total"
        :status="store.detail.status"
        class="progress"
      />
      <p v-if="store.polling" class="fallback-hint">SSE 连接不稳定，已切换为轮询刷新。</p>
      <BatchTable :items="store.detail.items" :running="store.itemRunning" />
      <div v-if="store.finished" class="report">
        <el-link
          v-if="store.detail.report_url"
          type="primary"
          :href="reportUrl(store.batchId)"
          download
        >
          下载 report.csv
        </el-link>
      </div>
    </template>
  </div>
</template>

<style scoped>
.batch-view {
  max-width: 1080px;
}
.uploads {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.upload-box {
  flex: 1 1 280px;
}
.tip {
  padding: 16px;
  color: #606266;
}
.pair-hint {
  font-size: 13px;
  color: #909399;
  margin: 8px 0;
}
.actions {
  display: flex;
  gap: 12px;
  align-items: center;
  margin: 12px 0;
}
.note {
  max-width: 320px;
}
.alert {
  margin: 12px 0;
}
.progress {
  margin: 16px 0;
}
.fallback-hint {
  font-size: 12px;
  color: #e6a23c;
}
.report {
  margin-top: 12px;
}
</style>
