<script setup lang="ts">
defineProps<{ originalUrl: string; resultUrl: string }>()
</script>

<template>
  <div class="compare">
    <figure>
      <figcaption>原图</figcaption>
      <el-image :src="originalUrl" fit="contain" class="image" />
    </figure>
    <figure>
      <figcaption>修复结果</figcaption>
      <el-image :src="resultUrl" fit="contain" class="image" />
    </figure>
  </div>
</template>

<style scoped>
.compare {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
figure {
  margin: 0;
  flex: 1 1 280px;
}
figcaption {
  font-size: 13px;
  color: #606266;
  margin-bottom: 6px;
}
.image {
  width: 100%;
  max-height: 420px;
  border: 1px solid #ebeef5;
  background: #fafafa;
}
</style>
