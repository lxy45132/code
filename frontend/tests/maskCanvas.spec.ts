import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import MaskCanvas from '../src/components/MaskCanvas.vue'

// jsdom 无 canvas 2d 实现：用记录式 stub 验证画笔路径与工具语义。
function makeCtx() {
  return {
    drawImage: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    getImageData: vi.fn(() => ({ data: new Uint8ClampedArray([0, 0, 0, 255]) })),
    putImageData: vi.fn(),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    lineCap: '',
    lineJoin: '',
  }
}

class FakeImage {
  naturalWidth = 8
  naturalHeight = 8
  onload: (() => void) | null = null
  set src(_value: string) {
    queueMicrotask(() => this.onload?.())
  }
}

let ctx: ReturnType<typeof makeCtx>

beforeEach(() => {
  ctx = makeCtx()
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
    ctx as unknown as CanvasRenderingContext2D,
  )
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((cb: BlobCallback) => {
    cb(new Blob(['mask'], { type: 'image/png' }))
  })
  HTMLCanvasElement.prototype.setPointerCapture = vi.fn()
  vi.stubGlobal('Image', FakeImage)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

async function mountCanvas() {
  const wrapper = mount(MaskCanvas, {
    props: { imageUrl: 'blob:test' },
    global: {
      stubs: {
        'el-radio-group': true,
        'el-radio-button': true,
        'el-slider': true,
        'el-button': true,
      },
    },
  })
  await new Promise((resolve) => setTimeout(resolve, 0))
  await wrapper.vm.$nextTick()
  return wrapper
}

describe('MaskCanvas', () => {
  it('loads image and initialises black mask', async () => {
    const wrapper = await mountCanvas()
    expect(wrapper.vm.ready).toBe(true)
    expect(ctx.drawImage).toHaveBeenCalled()
    expect(ctx.fillStyle).toBe('#000000')
    expect(ctx.fillRect).toHaveBeenCalledWith(0, 0, 8, 8)
  })

  it('brush paints white strokes along pointer path', async () => {
    const wrapper = await mountCanvas()
    const overlay = wrapper.find('canvas.overlay')
    await overlay.trigger('pointerdown', { clientX: 2, clientY: 2, pointerId: 1 })
    await overlay.trigger('pointermove', { clientX: 6, clientY: 4, pointerId: 1 })
    await overlay.trigger('pointerup', { pointerId: 1 })

    expect(ctx.strokeStyle).toBe('#ffffff')
    expect(ctx.moveTo).toHaveBeenCalledWith(2, 2)
    expect(ctx.lineTo).toHaveBeenCalledWith(6, 4)
    expect(ctx.stroke).toHaveBeenCalledTimes(2)
    expect(ctx.lineCap).toBe('round')
  })

  it('eraser paints black', async () => {
    const wrapper = await mountCanvas()
    wrapper.vm.tool = 'eraser'
    await wrapper.find('canvas.overlay').trigger('pointerdown', { clientX: 1, clientY: 1 })
    expect(ctx.strokeStyle).toBe('#000000')
  })

  it('clear fills black and invert flips pixels', async () => {
    const wrapper = await mountCanvas()
    wrapper.vm.clear()
    expect(ctx.fillRect).toHaveBeenLastCalledWith(0, 0, 8, 8)

    wrapper.vm.invert()
    expect(ctx.getImageData).toHaveBeenCalledWith(0, 0, 8, 8)
    const flipped = ctx.putImageData.mock.calls[0][0] as ImageData
    expect(Array.from(flipped.data.slice(0, 3))).toEqual([255, 255, 255])
  })

  it('exports mask as PNG blob', async () => {
    const wrapper = await mountCanvas()
    const blob = await wrapper.vm.getMaskBlob()
    expect(blob).toBeInstanceOf(Blob)
    expect(blob.type).toBe('image/png')
  })
})
