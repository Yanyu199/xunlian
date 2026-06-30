<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { getHealth, getTrainingDefaults, getTrainingStatus, predictFile, startTraining, uploadForQc, validateModel } from './api'

const health = ref(null)
const selectedFile = ref(null)
const validationModelFile = ref(null)
const validationScalerFile = ref(null)
const qc = ref(null)
const params = ref({})
const gpu = ref(null)
const job = ref(null)
const prediction = ref(null)
const modelValidation = ref(null)
const error = ref('')
const busy = ref(false)
const predicting = ref(false)
const validating = ref(false)
const activeTip = ref(null)
let timer = null

const parameterFields = [
  {
    key: 'layer_num',
    label: '地层层数',
    type: 'number',
    min: 2,
    tip: '反演模型假设的地下分层数。应结合钻孔揭露、煤岩层结构和目标精度设置，过少会欠拟合，过多会增加不适定性。'
  },
  {
    key: 'sample_size',
    label: '样本数量',
    type: 'number',
    min: 2,
    tip: '用于正演生成训练集的模型数量。数量越大泛化越稳，但训练和正演耗时越长；现场初调可小，正式训练应增大。'
  },
  {
    key: 'time_channels',
    label: '时间道数',
    type: 'number',
    min: 5,
    tip: '每条 Z 响应曲线参与训练的时间采样点数。通常应与实测文件时间道一致，减少会丢失晚期或早期信息。'
  },
  {
    key: 'time_min',
    label: '最小时间 s',
    type: 'number',
    step: '0.000001',
    tip: '训练使用的最早时间窗，单位为秒。应避开施工强干扰和关断异常段，同时保留浅部敏感信息。'
  },
  {
    key: 'time_max',
    label: '最大时间 s',
    type: 'number',
    step: '0.000001',
    tip: '训练使用的最晚时间窗，单位为秒。晚期数据对应更深部信息，但信噪比差时不宜盲目拉长。'
  },
  {
    key: 'use_prior',
    label: '使用先验估计',
    type: 'select',
    options: [
      { value: true, label: '是' },
      { value: false, label: '否' }
    ],
    tip: '是否根据上传 Z 数据先估计电阻率范围。现场数据建议开启，可减少无效样本并提高训练针对性。'
  },
  {
    key: 'r_min',
    label: '电阻率下限 Ω·m',
    type: 'number',
    min: 0.1,
    tip: '训练样本允许的最小电阻率。应覆盖含水、破碎或低阻异常体，过低会扩大搜索空间并降低稳定性。'
  },
  {
    key: 'r_max',
    label: '电阻率上限 Ω·m',
    type: 'number',
    min: 1,
    tip: '训练样本允许的最大电阻率。应覆盖完整煤岩层和高阻围岩，需大于下限，过大会增加反演多解性。'
  },
  {
    key: 'thickness_min',
    label: '厚度下限 m',
    type: 'number',
    min: 0.1,
    tip: '单层厚度的最小取值。应接近工程可分辨厚度，过小会生成现场数据难以约束的薄层模型。'
  },
  {
    key: 'thickness_max',
    label: '厚度上限 m',
    type: 'number',
    min: 1,
    tip: '单层厚度的最大取值。应与钻孔探测深度和地层尺度匹配，过大会削弱近孔方向分辨率。'
  },
  {
    key: 'epochs',
    label: '训练轮数 epochs',
    type: 'number',
    min: 1,
    tip: '神经网络完整遍历训练集的次数。损失仍下降可增加；验证损失不降或震荡时应减少或调学习率。'
  },
  {
    key: 'batch_size',
    label: '批大小 batch',
    type: 'number',
    min: 1,
    tip: '每次参数更新使用的样本数。GPU 显存 6GB 建议先用 32-64，显存充足再增大以提升吞吐。'
  },
  {
    key: 'learning_rate',
    label: '学习率',
    type: 'number',
    step: '0.0001',
    tip: '优化器每步更新幅度。过大会震荡或发散，过小会训练很慢；通常从 1e-3 到 1e-4 试起。'
  },
  {
    key: 'valid_portion',
    label: '验证集比例',
    type: 'number',
    step: '0.05',
    min: 0,
    max: 0.8,
    tip: '从样本中留作验证的数据比例。用于判断泛化能力，常用 0.1-0.2；样本很少时不宜过高。'
  },
  {
    key: 'device',
    label: '训练设备',
    type: 'select',
    options: [
      { value: 'auto', label: '自动优先 GPU' },
      { value: 'cuda', label: '强制 CUDA' },
      { value: 'cpu', label: 'CPU' }
    ],
    tip: '选择模型训练设备。auto 会在 CUDA 可用时用 GPU；强制 CUDA 可及时暴露 GPU 环境安装问题。'
  },
  {
    key: 'use_amp',
    label: 'CUDA 混合精度',
    type: 'select',
    options: [
      { value: true, label: '开启' },
      { value: false, label: '关闭' }
    ],
    tip: 'GPU 训练时使用半精度加速并降低显存占用。多数 NVIDIA GPU 建议开启；若损失异常再关闭排查。'
  },
  {
    key: 'torch_threads',
    label: 'CPU 线程数 0=自动',
    type: 'number',
    min: 0,
    tip: 'PyTorch 在 CPU 上使用的计算线程数。GPU 训练一般设 0 自动；CPU 训练可按服务器核心数适当设置。'
  },
  {
    key: 'stall_seconds',
    label: '卡住阈值 s',
    type: 'number',
    min: 10,
    tip: '超过该秒数没有进度更新即提示可能卡住。正演或先验搜索较慢时可适当调大，避免误报。'
  },
  {
    key: 'prior_init_points',
    label: '先验随机点',
    type: 'number',
    min: 0,
    tip: '先验范围搜索的初始随机模型数。越大越容易找到合理范围，但会明显增加正演耗时。'
  },
  {
    key: 'prior_iter',
    label: '先验迭代',
    type: 'number',
    min: 0,
    tip: '先验范围优化迭代次数。用于细化电阻率范围，现场快速试算可降低，正式训练再提高。'
  },
  {
    key: 'prior_sim_samples',
    label: '先验模拟样本',
    type: 'number',
    min: 2,
    tip: '每轮先验评估中参与对比的模拟样本数。应与实测曲线数量同量级，过大时会拖慢自动识别。'
  },
  {
    key: 'forward_batch_size',
    label: '正演批大小',
    type: 'number',
    min: 1,
    tip: '生成训练样本时每批正演模型数量。越大越快但占用内存更高；内存或显存紧张时应降低。'
  }
]

const statusClass = computed(() => health.value?.status === 'ok' ? 'ok' : 'warn')
const canTrain = computed(() => selectedFile.value && !['queued', 'running'].includes(job.value?.status))
const canPredict = computed(() => (
  selectedFile.value &&
  health.value?.model_exists &&
  health.value?.scaler_exists &&
  !busy.value &&
  !predicting.value &&
  !['queued', 'running'].includes(job.value?.status)
))
const canValidate = computed(() => (
  selectedFile.value &&
  validationModelFile.value &&
  (validationScalerFile.value || health.value?.scaler_exists) &&
  !busy.value &&
  !predicting.value &&
  !validating.value &&
  !['queued', 'running'].includes(job.value?.status)
))

const trainStatusText = computed(() => {
  const status = job.value?.status
  if (!status) return '未开始'
  return { queued: '排队中', running: '训练中', completed: '已完成', failed: '失败' }[status] || status
})

const stageBars = computed(() => {
  const names = ['数据读取', '先验范围', '正演样本', '训练准备', '模型训练', '保存结果']
  const current = job.value?.stage_index || 0
  return names.map((name, index) => {
    const stageIndex = index + 1
    let progress = 0
    if (stageIndex < current) progress = 100
    if (stageIndex === current) progress = job.value?.stage_progress || 0
    if (job.value?.status === 'completed') progress = 100
    return { name, progress, active: stageIndex === current }
  })
})

const logRows = computed(() => job.value?.logs || [])
const predictionRows = computed(() => prediction.value?.results || [])
const validationChecks = computed(() => modelValidation.value?.checks || [])

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  const number = Number(value)
  if (Math.abs(number) >= 1000 || Math.abs(number) < 0.01) return number.toExponential(2)
  return number.toFixed(digits)
}

function formatSeconds(value) {
  if (value === null || value === undefined) return '-'
  const seconds = Math.max(0, Math.round(value))
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h) return `${h}小时${m}分`
  if (m) return `${m}分${s}秒`
  return `${s}秒`
}

function normalizeSelectValue(value) {
  if (value === true) return 'true'
  if (value === false) return 'false'
  return value
}

function updateParam(key, value) {
  const current = params.value[key]
  if (typeof current === 'boolean') params.value[key] = Boolean(value)
  else if (typeof current === 'number') params.value[key] = Number(value)
  else params.value[key] = value
}

function onFieldChange(field, value) {
  if (field.type === 'select') {
    const option = field.options.find((item) => String(item.value) === String(value))
    updateParam(field.key, option ? option.value : value)
    return
  }
  updateParam(field.key, value)
}

function showTip(field, event) {
  activeTip.value = {
    key: field.key,
    label: field.label,
    text: field.tip,
    x: event.clientX + 14,
    y: event.clientY + 14
  }
}

function moveTip(event) {
  if (!activeTip.value) return
  activeTip.value = {
    ...activeTip.value,
    x: event.clientX + 14,
    y: event.clientY + 14
  }
}

function hideTip() {
  activeTip.value = null
}

function onFileChange(event) {
  selectedFile.value = event.target.files?.[0] || null
  qc.value = null
  job.value = null
  prediction.value = null
  modelValidation.value = null
  error.value = ''
  if (selectedFile.value) runQc(true)
}

function onValidationModelChange(event) {
  validationModelFile.value = event.target.files?.[0] || null
  modelValidation.value = null
  error.value = ''
}

function onValidationScalerChange(event) {
  validationScalerFile.value = event.target.files?.[0] || null
  modelValidation.value = null
  error.value = ''
}

async function refreshHealth() {
  try {
    health.value = await getHealth()
    gpu.value = health.value?.gpu || gpu.value
  } catch (err) {
    error.value = err.message
  }
}

async function loadDefaults() {
  try {
    const defaults = await getTrainingDefaults()
    gpu.value = defaults.gpu || null
    delete defaults.gpu
    params.value = defaults
  } catch (err) {
    error.value = err.message
  }
}

async function runQc(applySuggestions = true) {
  if (!selectedFile.value) return
  busy.value = true
  error.value = ''
  try {
    qc.value = await uploadForQc(selectedFile.value)
    if (applySuggestions && qc.value?.suggested_params) {
      params.value = { ...params.value, ...qc.value.suggested_params }
    }
  } catch (err) {
    error.value = err.message
  } finally {
    busy.value = false
  }
}

async function pollJob(jobId) {
  try {
    job.value = await getTrainingStatus(jobId)
    if (['completed', 'failed'].includes(job.value.status)) {
      stopPolling()
      await refreshHealth()
    }
  } catch (err) {
    error.value = err.message
    stopPolling()
  }
}

function stopPolling() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

async function runTraining() {
  if (!selectedFile.value) return
  busy.value = true
  error.value = ''
  job.value = null
  stopPolling()
  try {
    const started = await startTraining(selectedFile.value, params.value)
    job.value = { job_id: started.job_id, status: started.status, total_progress: 0, stage_progress: 0 }
    await pollJob(started.job_id)
    timer = setInterval(() => pollJob(started.job_id), 1200)
  } catch (err) {
    error.value = err.message
  } finally {
    busy.value = false
  }
}

async function runPrediction() {
  if (!selectedFile.value) return
  predicting.value = true
  error.value = ''
  prediction.value = null
  try {
    prediction.value = await predictFile(selectedFile.value)
  } catch (err) {
    error.value = err.message
  } finally {
    predicting.value = false
  }
}

async function runModelValidation() {
  if (!selectedFile.value || !validationModelFile.value) return
  validating.value = true
  error.value = ''
  modelValidation.value = null
  try {
    modelValidation.value = await validateModel(validationModelFile.value, selectedFile.value, validationScalerFile.value)
  } catch (err) {
    error.value = err.message
  } finally {
    validating.value = false
  }
}

onMounted(async () => {
  await Promise.all([refreshHealth(), loadDefaults()])
})
onUnmounted(stopPolling)
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div>
        <h1>孔中瞬变电磁训练工作台</h1>
        <p>原始 z 数据训练 · 自动参数识别 · GPU 优先</p>
      </div>
      <button class="icon-button" title="刷新服务状态" @click="refreshHealth">↻</button>
    </header>

    <section class="status-strip">
      <div>
        <span class="label">后端</span>
        <strong :class="statusClass">{{ health?.status || '未连接' }}</strong>
      </div>
      <div>
        <span class="label">GPU</span>
        <strong :class="gpu?.cuda_available ? 'ok' : 'warn'">
          {{ gpu?.cuda_available ? `${gpu.device_name}` : '未检测到 CUDA' }}
        </strong>
      </div>
      <div>
        <span class="label">正演</span>
        <strong :class="health?.forward_backend?.gpu_accelerated ? 'ok' : 'warn'">
          {{ health?.forward_backend?.gpu_accelerated ? `GPU/CuPy ${health.forward_backend.device_name || ''}` : 'CPU/NumPy' }}
        </strong>
      </div>
      <div>
        <span class="label">训练状态</span>
        <strong :class="job?.status === 'failed' ? 'warn' : 'ok'">{{ trainStatusText }}</strong>
      </div>
    </section>

    <section class="workspace">
      <aside class="side-panel">
        <div class="file-box">
          <label for="file">原始 z 数据文件</label>
          <input id="file" type="file" accept=".txt,.csv,.dat" @change="onFileChange" />
          <div class="file-name">{{ selectedFile?.name || '未选择文件' }}</div>
        </div>

        <div class="file-box validation-upload">
          <label for="model-file">验证模型 .pt 文件</label>
          <input id="model-file" type="file" accept=".pt,.pth" @change="onValidationModelChange" />
          <div class="file-name">{{ validationModelFile?.name || '未选择模型文件' }}</div>
        </div>

        <div class="file-box validation-upload">
          <label for="scaler-file">对应 scaler .json（可选）</label>
          <input id="scaler-file" type="file" accept=".json" @change="onValidationScalerChange" />
          <div class="file-name">{{ validationScalerFile?.name || '未选择时使用当前激活 scaler' }}</div>
        </div>

        <div class="actions">
          <button :disabled="!selectedFile || busy" @click="runQc(true)">质控/自动识别</button>
          <button class="primary" :disabled="!canTrain || busy" @click="runTraining">训练</button>
          <button :disabled="!canPredict" @click="runPrediction">反演预测</button>
          <button :disabled="!canValidate" @click="runModelValidation">模型验证</button>
        </div>

        <div v-if="error" class="alert">{{ error }}</div>
        <div v-if="selectedFile && !health?.model_exists" class="alert soft">
          当前还没有激活模型。请先训练一次，训练完成后系统会自动激活最新模型。
        </div>
        <div v-if="job?.stalled" class="alert">
          训练超过 {{ job.params?.stall_seconds }} 秒没有进度更新，可能卡住。请检查后端控制台、GPU/CPU 占用，或降低样本数、先验搜索次数、epoch。
        </div>
      </aside>

      <section class="main-panel">
        <div class="panel-head">
          <h2>机器学习参数</h2>
          <span>上传文件后会自动填入可识别参数，训练前仍可手动调整</span>
        </div>

        <div class="param-grid">
          <label v-for="field in parameterFields" :key="field.key" class="param-field">
            <span class="param-label">
              {{ field.label }}
              <button
                class="help-icon"
                type="button"
                :aria-label="`${field.label} 参数说明`"
                @mouseenter="showTip(field, $event)"
                @mousemove="moveTip"
                @mouseleave="hideTip"
                @focus="showTip(field, $event)"
                @blur="hideTip"
              >
                ?
              </button>
            </span>
            <select
              v-if="field.type === 'select'"
              :value="normalizeSelectValue(params[field.key])"
              @change="onFieldChange(field, $event.target.value)"
            >
              <option v-for="option in field.options" :key="String(option.value)" :value="normalizeSelectValue(option.value)">
                {{ option.label }}
              </option>
            </select>
            <input
              v-else
              :type="field.type"
              :step="field.step"
              :min="field.min"
              :max="field.max"
              :value="params[field.key]"
              @input="onFieldChange(field, $event.target.value)"
            />
          </label>
        </div>

        <div class="panel-head result-head">
          <h2>训练进度</h2>
          <span v-if="job">任务 {{ job.job_id }}</span>
        </div>

        <div v-if="job" class="progress-area">
          <div class="progress-row">
            <span>总进度</span>
            <div class="progress"><i :style="{ width: `${job.total_progress || 0}%` }"></i></div>
            <strong>{{ formatNumber(job.total_progress || 0, 1) }}%</strong>
          </div>
          <div class="stage-list">
            <div v-for="stage in stageBars" :key="stage.name" :class="['stage-item', { active: stage.active }]">
              <span>{{ stage.name }}</span>
              <div class="progress"><i :style="{ width: `${stage.progress}%` }"></i></div>
              <strong>{{ formatNumber(stage.progress, 1) }}%</strong>
            </div>
          </div>
          <div class="metrics">
            <div><span>当前步骤</span><strong>{{ job.stage_index }}/{{ job.stage_count }}</strong></div>
            <div><span>已用时间</span><strong>{{ formatSeconds(job.elapsed_seconds) }}</strong></div>
            <div><span>预计剩余</span><strong>{{ formatSeconds(job.eta_seconds) }}</strong></div>
            <div><span>最近更新</span><strong>{{ formatSeconds(job.last_update_seconds_ago) }}前</strong></div>
          </div>
          <p class="message">{{ job.message }}</p>
        </div>
        <div v-else class="empty">尚未开始训练</div>

        <div v-if="qc" class="panel-head result-head">
          <h2>数据质控与自动识别</h2>
          <span :class="['pill', qc.status]">{{ qc.status }}</span>
        </div>
        <div v-if="qc" class="metrics">
          <div><span>测点数</span><strong>{{ qc.point_count }}</strong></div>
          <div><span>时间道</span><strong>{{ qc.time_count }}</strong></div>
          <div><span>起始时间</span><strong>{{ formatNumber(qc.time_min, 6) }}</strong></div>
          <div><span>终止时间</span><strong>{{ formatNumber(qc.time_max, 6) }}</strong></div>
        </div>
        <div v-if="qc?.metadata" class="result-box">
          <h2>自动识别结果</h2>
          <p><strong>数据格式：</strong>{{ qc.metadata.format_label || qc.metadata.format }}</p>
          <p><strong>时间单位推断：</strong>{{ qc.metadata.time_unit_inferred }}</p>
          <p v-if="qc.metadata.point_id_columns"><strong>测点标识列：</strong>第 {{ qc.metadata.point_id_columns.join('、') }} 列</p>
          <p v-if="qc.metadata.time_source_column"><strong>时间列：</strong>第 {{ qc.metadata.time_source_column }} 列</p>
          <p v-if="qc.metadata.response_source_column"><strong>响应数据列：</strong>第 {{ qc.metadata.response_source_column }} 列</p>
          <p v-if="qc.metadata.raw_time_min"><strong>原始时间范围：</strong>{{ formatNumber(qc.metadata.raw_time_min, 3) }} - {{ formatNumber(qc.metadata.raw_time_max, 3) }}</p>
          <p><strong>响应范围：</strong>{{ formatNumber(qc.response_min, 6) }} - {{ formatNumber(qc.response_max, 6) }}</p>
        </div>

        <div v-if="job?.result" class="result-box key-result">
          <h2>训练输出</h2>
          <p><strong>模型文件：</strong>{{ job.result.model_path }}</p>
          <p><strong>归一化文件：</strong>{{ job.result.scaler_path }}</p>
          <p v-if="job.result.active_model_path"><strong>当前激活模型：</strong>{{ job.result.active_model_path }}</p>
          <p><strong>训练历史：</strong>{{ job.result.history_path }}</p>
          <p><strong>使用设备：</strong>{{ job.result.used_device }}</p>
          <p><strong>最佳损失：</strong>{{ formatNumber(job.result.best_loss, 6) }}</p>
          <p><strong>电阻率范围：</strong>{{ formatNumber(job.result.resistivity_range?.[0]) }} - {{ formatNumber(job.result.resistivity_range?.[1]) }} Ω·m</p>
        </div>

        <div v-if="predicting" class="empty">正在执行反演预测</div>
        <div v-if="prediction" class="panel-head result-head">
          <h2>反演预测结果</h2>
          <span>模型时间道 {{ prediction.model?.time_channels }}，层数 {{ prediction.model?.layer_num }}</span>
        </div>
        <div v-if="prediction?.warnings?.length" class="alert soft">
          <p v-for="warning in prediction.warnings" :key="warning">{{ warning }}</p>
        </div>
        <div v-if="predictionRows.length" class="result-table">
          <table>
            <thead>
              <tr>
                <th>测点</th>
                <th>质控</th>
                <th v-for="layer in predictionRows[0].layers" :key="layer.layer">
                  第 {{ layer.layer }} 层
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in predictionRows" :key="row.point">
                <td>{{ row.point }}</td>
                <td>{{ row.qc_status }}</td>
                <td v-for="layer in row.layers" :key="layer.layer">
                  <strong>{{ formatNumber(layer.resistivity, 2) }} Ω·m</strong>
                  <span v-if="layer.thickness !== null">{{ formatNumber(layer.thickness, 2) }} m</span>
                  <span v-else>半空间</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-if="validating" class="empty">正在验证模型</div>
        <div v-if="modelValidation" class="panel-head result-head">
          <h2>模型验证结果</h2>
          <span :class="['pill', modelValidation.status]">
            {{ modelValidation.status }} · {{ formatNumber(modelValidation.score, 1) }} 分
          </span>
        </div>
        <div v-if="modelValidation" class="metrics">
          <div><span>输入超界比例</span><strong>{{ formatNumber((modelValidation.summary?.input_outside_ratio || 0) * 100, 1) }}%</strong></div>
          <div><span>输出最小值</span><strong>{{ formatNumber(modelValidation.summary?.prediction_stats?.min, 3) }}</strong></div>
          <div><span>输出最大值</span><strong>{{ formatNumber(modelValidation.summary?.prediction_stats?.max, 3) }}</strong></div>
          <div><span>模型时间道</span><strong>{{ modelValidation.model?.time_channels }}</strong></div>
        </div>
        <div v-if="modelValidation?.warnings?.length" class="alert soft">
          <p v-for="warning in modelValidation.warnings" :key="warning">{{ warning }}</p>
        </div>
        <div v-if="validationChecks.length" class="validation-list">
          <div v-for="check in validationChecks" :key="check.name" :class="['validation-item', check.status]">
            <strong>{{ check.name }}</strong>
            <span>{{ check.message }}</span>
          </div>
        </div>

        <div v-if="logRows.length" class="log-box">
          <div v-for="(line, index) in logRows" :key="index" :class="['log-line', line.level, { key: line.key }]">
            <span class="log-time">{{ line.time }}</span>
            <span class="log-stage">{{ line.stage }}</span>
            <span class="log-message">{{ line.message }}</span>
          </div>
        </div>
      </section>
    </section>

    <teleport to="body">
      <div
        v-if="activeTip"
        class="parameter-tooltip"
        role="tooltip"
        :style="{ left: `${activeTip.x}px`, top: `${activeTip.y}px` }"
      >
        <strong>{{ activeTip.label }}</strong>
        <span>{{ activeTip.text }}</span>
      </div>
    </teleport>
  </main>
</template>
