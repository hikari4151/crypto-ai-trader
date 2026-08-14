# AI 模型自动拉取（Cherry Studio 式配置）设计

日期：2026-08-14
状态：已确认

## 目标

用户填入 API Key + Base URL 后，可一键/保存时自动拉取该服务支持的模型列表并填充下拉，
无需手动查文档填模型名。交互对齐 Cherry Studio。

## 范围

- 后端：`AIClient.list_models(base_url, api_key)` + `POST /api/ai/models` 端点
- 前端：AI 设置页「获取模型」按钮 + 保存成功后自动拉取
- 不保存表单中的 api_key/base_url（与 `/test` 同模式，只读使用）

## 接口

`GET {base_url}/models`（OpenAI 兼容标准；OpenAI/DeepSeek/自建网关均支持）
- 成功：解析 `data[].id`，去重排序，返回 `{ok, models: [id...]}`
- 401：提示 API Key 无效
- 404：提示服务不支持 /models 端点，保留手动输入
- 网络错误：友好提示

## 前端交互

- 模型输入框旁「获取模型」按钮：拉取 → 填充 datalist → 自动选中第一个（当前模型不在列表时）
- `saveAiSettings` 成功后异步拉取一次（不覆盖用户已填模型，仅当模型为空或不在列表时填第一个）
- 模型框保持可手动输入（无 models 端点的服务兜底）

## 安全

- api_key 仅用于该次请求，不落库、不写日志、不返回给前端

## 验证

- 本地 mock `/models` 服务全链路测试端点（成功/401/404）
- 编译 + 服务器冒烟 + 前端 JS 语法
