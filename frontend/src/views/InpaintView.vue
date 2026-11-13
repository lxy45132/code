<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage, type UploadFile } from 'element-plus'
import MaskCanvas from '../components/MaskCanvas.vue'
import ImageCompare from '../components/ImageCompare.vue'
import { useInpaintStore } from '../stores/inpaint'

const store = useInpaintStore()
const maskCanvas = ref<InstanceType<typeof MaskCanvas> | null>(null)

function onFileChange(uploadFile: UploadFile) {
  const raw = uploadFile.raw
  if (!raw) return
  if (!raw.type.startsWith('image/')) {
    ElMessage.error('请选择图片文件')
    return
  }
  store.setImage(raw)
}

async function submit() {
  if (!store.imageFile) {
    ElMessage.warning('请先选择图片')
    return
  }
  if (!maskCanvas.value) return
  try {
    const mask = await maskCanvas.value.getMaskBlob()
    await store.submit(mask)
    if (store.result) ElMessage.success('修复完成')
  } catch {
    ElMessage.error('掩码导出失败')
  }
}
</script>

<template>
  <div class="inpaint-view">
    <h2>单图修复</h2>
    <el-upload
      drag
      :auto-upload="false"
      :show-file-list="false"
      accept="image/*"
      :on-change="onFileChange"
      class="uploader"
    >
      <div class="upload-tip">将图片拖到此处，或<em>点击选择</em>（≤20MB，边长≤4096）</div>
    </el-upload>

    <template v-if="store.imageUrl">
      <MaskCanvas ref="maskCanvas" :image-url="store.imageUrl" />
      <div class="actions">
        <el-input v-model="store.note" placeholder="备注（可选）" class="note" maxlength="200" />
        <el-button type="primary" :loading="store.loading" @click="submit"> 开始修复 </el-button>
        <el-button @click="store.reset()"> 重置 </el-button>
      </div>
    </template>

    <el-alert v-if="store.error" :title="store.error" type="error" show-icon class="alert" />

    <template v-if="store.result">
      <ImageCompare :original-url="store.imageUrl" :result-url="store.result.output_url" />
      <el-descriptions :column="3" border class="meta">
        <el-descriptions-item label="耗时">
          {{ store.result.latency_ms.toFixed(1) }} ms
        </el-descriptions-item>
        <el-descriptions-item label="掩码比例">
          {{ (store.result.mask_ratio * 100).toFixed(2) }}%
        </el-descriptions-item>
        <el-descriptions-item label="模型版本">
          {{ store.result.model_version }}
        </el-descriptions-item>
      </el-descriptions>
      <el-alert
        v-for="warning in store.result.warnings"
        :key="warning"
        :title="warning"
        type="warning"
        show-icon
        class="alert"
      />
      <el-link type="primary" :href="store.result.output_url" download class="download">
        下载结果 PNG
      </el-link>
    </template>
  </div>
</template>

<style scoped>
.inpaint-view {
  max-width: 960px;
}
.uploader {
  margin-bottom: 16px;
}
.upload-tip {
  padding: 12px;
  color: #606266;
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
.meta {
  margin: 16px 0;
}
.download {
  margin-top: 8px;
}
</style>
