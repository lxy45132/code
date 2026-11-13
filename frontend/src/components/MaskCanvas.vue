<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'

// 掩码语义与后端一致：白色（非纯黑）= 洞。底图 + 半透明掩码双层 canvas。
const props = defineProps<{ imageUrl: string }>()

const baseCanvas = ref<HTMLCanvasElement | null>(null)
const maskCanvas = ref<HTMLCanvasElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const tool = ref<'brush' | 'eraser'>('brush')
const brushSize = ref(24)
const ready = ref(false)

let baseCtx: CanvasRenderingContext2D | null = null
let maskCtx: CanvasRenderingContext2D | null = null
let drawing = false
let lastPoint: { x: number; y: number } | null = null

function loadImage() {
  ready.value = false
  if (!props.imageUrl || !baseCanvas.value || !maskCanvas.value) return
  const image = new Image()
  image.onload = () => {
    const base = baseCanvas.value
    const mask = maskCanvas.value
    if (!base || !mask) return
    base.width = image.naturalWidth
    base.height = image.naturalHeight
    mask.width = image.naturalWidth
    mask.height = image.naturalHeight
    baseCtx = base.getContext('2d')
    maskCtx = mask.getContext('2d')
    baseCtx?.drawImage(image, 0, 0)
    if (maskCtx) {
      maskCtx.fillStyle = '#000000'
      maskCtx.fillRect(0, 0, mask.width, mask.height)
    }
    ready.value = true
  }
  image.src = props.imageUrl
}

onMounted(loadImage)
watch(() => props.imageUrl, loadImage)

function canvasPoint(event: PointerEvent): { x: number; y: number } {
  const canvas = maskCanvas.value
  if (!canvas) return { x: 0, y: 0 }
  const rect = canvas.getBoundingClientRect()
  const scaleX = rect.width ? canvas.width / rect.width : 1
  const scaleY = rect.height ? canvas.height / rect.height : 1
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  }
}

function strokeTo(point: { x: number; y: number }) {
  if (!maskCtx) return
  maskCtx.strokeStyle = tool.value === 'brush' ? '#ffffff' : '#000000'
  maskCtx.lineWidth = brushSize.value
  maskCtx.lineCap = 'round'
  maskCtx.lineJoin = 'round'
  maskCtx.beginPath()
  const from = lastPoint ?? point
  maskCtx.moveTo(from.x, from.y)
  maskCtx.lineTo(point.x, point.y)
  maskCtx.stroke()
  lastPoint = point
}

function onPointerDown(event: PointerEvent) {
  if (!ready.value) return
  drawing = true
  lastPoint = null
  maskCanvas.value?.setPointerCapture(event.pointerId)
  strokeTo(canvasPoint(event))
}

function onPointerMove(event: PointerEvent) {
  if (drawing) strokeTo(canvasPoint(event))
}

function onPointerUp() {
  drawing = false
  lastPoint = null
}

function clear() {
  const canvas = maskCanvas.value
  if (!maskCtx || !canvas) return
  maskCtx.fillStyle = '#000000'
  maskCtx.fillRect(0, 0, canvas.width, canvas.height)
}

function invert() {
  const canvas = maskCanvas.value
  if (!maskCtx || !canvas) return
  const data = maskCtx.getImageData(0, 0, canvas.width, canvas.height)
  for (let i = 0; i < data.data.length; i += 4) {
    data.data[i] = 255 - data.data[i]
    data.data[i + 1] = 255 - data.data[i + 1]
    data.data[i + 2] = 255 - data.data[i + 2]
  }
  maskCtx.putImageData(data, 0, 0)
}

function pickMaskFile() {
  fileInput.value?.click()
}

async function importMask(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  const canvas = maskCanvas.value
  if (!file || !maskCtx || !canvas) return
  const url = URL.createObjectURL(file)
  const image = new Image()
  image.onload = () => {
    maskCtx?.drawImage(image, 0, 0, canvas.width, canvas.height)
    URL.revokeObjectURL(url)
  }
  image.src = url
  input.value = ''
}

function getMaskBlob(): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const canvas = maskCanvas.value
    if (!canvas) {
      reject(new Error('掩码画布未就绪'))
      return
    }
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('掩码导出失败'))
    }, 'image/png')
  })
}

defineExpose({ getMaskBlob, clear, invert, importMask, tool, brushSize, ready })
</script>

<template>
  <div class="mask-canvas">
    <div class="toolbar">
      <el-radio-group v-model="tool" size="small">
        <el-radio-button value="brush"> 画笔（涂洞） </el-radio-button>
        <el-radio-button value="eraser"> 橡皮 </el-radio-button>
      </el-radio-group>
      <div class="size">
        <span>笔刷</span>
        <el-slider v-model="brushSize" :min="2" :max="128" :show-tooltip="false" />
      </div>
      <el-button size="small" @click="clear"> 清除 </el-button>
      <el-button size="small" @click="invert"> 反转 </el-button>
      <el-button size="small" @click="pickMaskFile"> 导入掩码 </el-button>
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        class="hidden-input"
        @change="importMask"
      />
    </div>
    <div class="stack">
      <canvas ref="baseCanvas" class="layer" />
      <canvas
        ref="maskCanvas"
        class="layer overlay"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
        @pointerleave="onPointerUp"
      />
    </div>
    <p class="hint">白色区域为待修复的洞；支持导入已有掩码（非纯黑即洞）。</p>
  </div>
</template>

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.size {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 180px;
  font-size: 12px;
  color: #606266;
}
.hidden-input {
  display: none;
}
.stack {
  position: relative;
  display: inline-block;
  max-width: 100%;
  border: 1px solid #dcdfe6;
  line-height: 0;
}
.layer {
  max-width: 100%;
  height: auto;
}
.overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0.5;
  cursor: crosshair;
  touch-action: none;
}
.hint {
  font-size: 12px;
  color: #909399;
  margin-top: 6px;
}
</style>
