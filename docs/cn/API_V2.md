# API v2 接口说明

本文档说明 Jetson 场景下新增的 HTTP 接口约定，重点覆盖 `/v1/audio/speech` 和 `/v1/voices/metadata`。

## 1. `/v1/audio/speech`

兼容 OpenAI 风格接口，同时支持两种调用模式：

- 旧模式：传 `voice` 或 `reference_id`
- 新模式：直接传参考音频元数据，不依赖本地 voice registry

### 1.1 旧模式

请求体示例：

```json
{
  "model": "gpt-sovits-v2proplus-jetson",
  "input": "这是一个旧模式示例。",
  "voice": "voice-123",
  "language": "zh",
  "response_format": "wav"
}
```

### 1.2 新模式

新模式下必须额外提供：

- `prompt_text`
- `prompt_lang`

参考音频支持以下 4 种输入方式，任选其一：

- `ref_audio`
- `ref_audio_path`
- `ref_audio_base64`
- `ref_audio_file`

#### 方式 A：传 URL 或本地路径

```json
{
  "model": "gpt-sovits-v2proplus-jetson",
  "input": "这是一个 URL 参考音频示例。",
  "language": "zh",
  "ref_audio": "https://example.com/reference.wav",
  "prompt_text": "今天天气很好，我们做一次测试。",
  "prompt_lang": "zh",
  "response_format": "wav"
}
```

说明：

- `ref_audio` 和 `ref_audio_path` 是等价字段
- 如果是 `http/https` URL，服务会先下载到临时文件再推理
- 推理结束后会自动清理临时文件

#### 方式 B：传 base64

```json
{
  "model": "gpt-sovits-v2proplus-jetson",
  "input": "这是一个 base64 参考音频示例。",
  "language": "zh",
  "ref_audio_base64": "UklGRiQAAABXQVZF...",
  "prompt_text": "今天天气很好，我们做一次测试。",
  "prompt_lang": "zh",
  "response_format": "wav"
}
```

说明：

- 支持纯 base64
- 也支持 data URL，例如 `data:audio/wav;base64,...`

#### 方式 C：`multipart/form-data` 上传二进制文件

```bash
curl -X POST http://127.0.0.1:9880/v1/audio/speech \
  -F model=gpt-sovits-v2proplus-jetson \
  -F input='这是一个文件上传示例。' \
  -F language=zh \
  -F prompt_text='今天天气很好，我们做一次测试。' \
  -F prompt_lang=zh \
  -F ref_audio_file=@reference.wav
```

说明：

- `ref_audio_file` 仅在 `multipart/form-data` 下可用
- 服务会将上传文件保存为临时文件，推理结束后自动清理

### 1.3 字段优先级

`/v1/audio/speech` 内部按以下优先级解析参考音频：

1. `ref_audio_file`
2. `ref_audio_base64`
3. `ref_audio` / `ref_audio_path`
4. `reference_id` / `voice`

### 1.4 错误约束

如果走新模式但缺少 `prompt_text` 或 `prompt_lang`，接口会返回 `400`。

如果未提供任何参考音频来源，接口会返回 `400`：

```text
voice, reference_id, ref_audio, ref_audio_path, ref_audio_base64, or ref_audio_file is required
```

## 2. `/v1/voices/metadata`

这个接口用于“准备 TTS 所需元数据”，不写入本地 voice registry，适合由外部 voice 服务统一管理音色记录和缓存。

请求方式二选一：

- 上传文件 `file`
- 传 `audio_url`

同时必须传：

- `prompt_text`
- `prompt_lang`

### 2.1 使用 `audio_url`

```bash
curl -X POST http://127.0.0.1:9880/v1/voices/metadata \
  -F audio_url=https://example.com/reference.wav \
  -F prompt_text='今天天气很好，我们做一次测试。' \
  -F prompt_lang=zh
```

### 2.2 上传文件

```bash
curl -X POST http://127.0.0.1:9880/v1/voices/metadata \
  -F prompt_text='今天天气很好，我们做一次测试。' \
  -F prompt_lang=zh \
  -F file=@reference.wav
```

返回示例：

```json
{
  "object": "voice_metadata",
  "source": "upload",
  "ref_audio": "/workspace/runtime/uploaded_references/xxx/reference.wav",
  "ref_audio_path": "/workspace/runtime/uploaded_references/xxx/reference.wav",
  "prompt_text": "今天天气很好，我们做一次测试。",
  "prompt_lang": "zh"
}
```

说明：

- `source=url` 时返回的是原始 URL
- `source=upload` 时返回的是本机保存后的路径
- 返回的 `ref_audio` / `ref_audio_path`、`prompt_text`、`prompt_lang` 可以直接转发给 `/v1/audio/speech`

## 3. 推荐架构

如果 voice 由独立服务统一管理，推荐这样使用：

1. 外部 voice 服务保存音色记录和缓存
2. 生成时由 voice 服务返回：
   `ref_audio` / `ref_audio_path`、`prompt_text`、`prompt_lang`
3. Jetson 只调用 `/v1/audio/speech` 做推理

这样可以保留旧的 `voice_id` 接口兼容性，同时让 Jetson 只承担生成逻辑。
