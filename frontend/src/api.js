const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(payload.detail || `请求失败: ${response.status}`)
  }
  return payload
}

export function getHealth() {
  return request('/api/health')
}

export function uploadForQc(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/api/data/qc', { method: 'POST', body: form })
}

export function getTrainingDefaults() {
  return request('/api/training/defaults')
}

export function startTraining(file, params) {
  const form = new FormData()
  form.append('file', file)
  form.append('params', JSON.stringify(params))
  return request('/api/training/start', { method: 'POST', body: form })
}

export function getTrainingStatus(jobId) {
  return request(`/api/training/status/${jobId}`)
}

export function predictFile(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/api/inversion/predict', { method: 'POST', body: form })
}
