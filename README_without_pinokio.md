# Alexandria Audiobook Generator — 不使用 Pinokio 的运行指南

> 本文档说明如何**完全不依赖 Pinokio** 在本机运行 Alexandria，Python 环境使用 **pyenv** 管理。
>
> Pinokio 只是启动器（创建虚拟环境、装依赖、抓取启动 URL），应用本体是标准的 FastAPI + Uvicorn 服务，`app/` 目录下的全部 Python 代码**没有任何 Pinokio 引用**。因此手动运行功能上零损失。
>
> 功能说明、Web UI 使用流程、音色/LoRA 详解请看主 [README.md](README.md) 与 [Wiki](https://github.com/Finrandojin/alexandria-audiobook/wiki)；本文只覆盖"如何把它跑起来"。

---

## 目录

- [Alexandria Audiobook Generator — 不使用 Pinokio 的运行指南](#alexandria-audiobook-generator--不使用-pinokio-的运行指南)
  - [目录](#目录)
  - [1. Pinokio 做了什么（对照表）](#1-pinokio-做了什么对照表)
  - [2. 前置要求](#2-前置要求)
    - [2.1 硬件与系统](#21-硬件与系统)
    - [2.2 系统软件](#22-系统软件)
    - [2.3 一个 LLM 服务](#23-一个-llm-服务)
  - [3. 安装 pyenv](#3-安装-pyenv)
    - [3.1 安装 pyenv](#31-安装-pyenv)
    - [3.2 编译依赖](#32-编译依赖)
    - [3.3 写入 shell 配置](#33-写入-shell-配置)
  - [4. 创建 Python 虚拟环境](#4-创建-python-虚拟环境)
    - [4.1 选择 Python 版本](#41-选择-python-版本)
    - [4.2 创建虚拟环境](#42-创建虚拟环境)
  - [5. 安装依赖](#5-安装依赖)
    - [5.1 Python 包（顺序与 `install.js` 一致）](#51-python-包顺序与-installjs-一致)
    - [5.2 安装 PyTorch（按平台二选一）](#52-安装-pytorch按平台二选一)
    - [5.3 可选：flash-attention（仅 NVIDIA）](#53-可选flash-attention仅-nvidia)
    - [5.4 验证安装](#54-验证安装)
  - [6. 启动](#6-启动)
    - [6.1 做成快捷命令（可选）](#61-做成快捷命令可选)
  - [7. 环境变量](#7-环境变量)
  - [8. 首次运行会发生什么](#8-首次运行会发生什么)
  - [9. 模型下载策略（external 模式）](#9-模型下载策略external-模式)
  - [10. 更新与重置](#10-更新与重置)
    - [10.1 更新（等价于 `update.js`）](#101-更新等价于-updatejs)
    - [10.2 重置（等价于 `reset.js`）](#102-重置等价于-resetjs)
  - [11. 故障排查](#11-故障排查)
  - [12. 可选方案：Docker / Colab](#12-可选方案docker--colab)
  - [附：一页速查](#附一页速查)

---

## 1. Pinokio 做了什么（对照表）

| Pinokio 脚本 | 作用 | 手动等价命令 |
|---|---|---|
| `install.js` | `uv cache clean` → 建 venv → `install -r requirements.txt` → `install qwen-tts==0.1.1` → 调用 `torch.js` | [第 5 节](#5-安装依赖) |
| `torch.js` | 按平台/GPU 装对应 torch 轮子（CUDA 12.8 / ROCm / CPU） | [第 5.2 节](#52-安装-pytorch按平台二选一) |
| `start.js` | 在 `app/` 下执行 `python app.py`，正则抓取 `http://...` 并把 URL 交给 UI | `python app/app.py` |
| `update.js` | `git pull` + 重跑 `install.js` | [第 10 节](#10-更新与重置) |
| `reset.js` | 删除生成产物（脚本、音色、分块、音频等） | [第 10 节](#10-更新与重置) |
| `pinokio.js` | 菜单渲染；用 `app/env` 是否存在判断"已安装" | 无（纯 UI） |

**唯一被 Pinokio 隐式提供的环境变量是 `ALEXANDRIA_HOST` / `ALEXANDRIA_PORT` / `ALEXANDRIA_CONFIG_PATH`，它们都有默认值**（见[第 7 节](#7-环境变量)），所以不设置也能跑。

---

## 2. 前置要求

### 2.1 硬件与系统

| 项目 | 要求 |
|---|---|
| **GPU** | 8 GB 显存起步，16 GB+ 推荐；CPU 也能跑但极慢 |
| **RAM** | 16 GB 推荐（最低 8 GB） |
| **磁盘** | ~20 GB（venv/PyTorch ~8 GB + 模型权重每个 ~3.5 GB + 音频工作区） |

平台支持情况（摘自主 README）：

| GPU | Windows | Linux | macOS |
|---|---|---|---|
| NVIDIA | ✅ 完整支持 | ✅ 完整支持 | — |
| AMD | ⚠️ 仅 CPU | ✅ ROCm 6.3+ | — |
| Apple Silicon | — | — | ⚠️ **仅 CPU**（不支持 MPS 加速） |
| Intel | — | — | ⚠️ 仅 CPU |

> **macOS 用户注意**：TTS 只能跑 CPU，速度比 GPU 慢一个数量级。如果只是想用 Web UI 和其他功能，建议把 TTS 指向远程服务器（[第 9 节](#9-模型下载策略external-模式)）。

### 2.2 系统软件

| 软件 | 用途 | 安装 |
|---|---|---|
| **ffmpeg** | MP3 编码（pydub）与 M4B 封装（直接调用） | 见下 |
| **git** | 拉取/更新代码 | 通常已自带 |
| **编译工具链** | pyenv 需要从源码编译 Python | 见 [3.2](#32-编译依赖) |

```bash
# macOS
brew install pyenv pyenv-virtualenv ffmpeg

# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# Fedora / RHEL
sudo dnf install -y ffmpeg-free
```

> ⚠️ **ffmpeg 是硬依赖**。缺少它时 MP3 导出会失败并**静默回退到 WAV**（`app/project.py` 有对应的检测与回退逻辑），M4B 导出则直接报错。用 `ffmpeg -version` 确认可用。

### 2.3 一个 LLM 服务

Alexandria **不含 LLM**，它通过 OpenAI 兼容 API 连接外部服务。生成剧本前必须有一个在跑：

| 服务 | 默认 Base URL | 说明 |
|---|---|---|
| [LM Studio](https://lmstudio.ai/) | `http://localhost:1234/v1` | 本地，推荐 Qwen3 系列 |
| [Ollama](https://ollama.ai/) | `http://localhost:11434/v1` | 本地，`ollama run qwen3` |
| [OpenAI](https://platform.openai.com/) / DeepSeek / OpenRouter | 官方地址 | 云端，需 API Key |

API Key 对本地服务随便填（例如 `local`）。

---

## 3. 安装 pyenv

### 3.1 安装 pyenv

```bash
# macOS（Homebrew，已含 pyenv-virtualenv）
brew install pyenv pyenv-virtualenv

# Linux（pyenv-installer，自带 pyenv-virtualenv）
curl -fsSL https://pyenv.run | bash
```

> **Windows**：pyenv 不支持原生 Windows。推荐装 **WSL2** 后走上面的 Linux 流程（CUDA 在 WSL2 中可正常使用）。若必须原生运行，请用 [pyenv-win](https://github.com/pyenv-win/pyenv-win)，并把[第 4.2 节](#42-创建虚拟环境)的虚拟环境改为 `python -m venv app/env`（pyenv-win 的 `pyenv virtualenv` 支持有限）。

### 3.2 编译依赖

pyenv 需要**从源码编译** Python，缺少这些库会导致 `pyenv install` 失败：

```bash
# macOS
brew install openssl readline sqlite3 xz zlib tcl-tk

# Ubuntu / Debian
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
  tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### 3.3 写入 shell 配置

**macOS + Homebrew** —— 路径由 Homebrew 接管，只需挂上初始化钩子：

```bash
{
  echo 'eval "$(pyenv init - zsh)"'
  echo 'eval "$(pyenv virtualenv-init -)"'
} >> ~/.zshrc
exec zsh
```

**Linux + pyenv-installer** —— 需要显式声明 `PYENV_ROOT`：

```bash
{
  echo 'export PYENV_ROOT="$HOME/.pyenv"'
  echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"'
  echo 'eval "$(pyenv init - bash)"'
  echo 'eval "$(pyenv virtualenv-init -)"'
} >> ~/.bashrc
exec bash
```

> 用 `pyenv root` 可确认实际的 `PYENV_ROOT`（Homebrew 一般是 `/opt/homebrew/var/pyenv` 或 `/usr/local/var/pyenv`）。

验证：

```bash
pyenv --version
pyenv versions
```

---

## 4. 创建 Python 虚拟环境

### 4.1 选择 Python 版本

**推荐 Python 3.10**，最低也是 3.10。依据：

- **最低 3.10**：代码使用了 `str | None` 类型语法（`app/generate_personas.py`），3.9 及以下无法运行；
- **推荐 3.10**：与 Pinokio 官方流程完全对齐 —— `torch.js` 提供的预编译轮子、以及可选的 flash-attention wheel 都是 **`cp310`**（[5.3](#53-可选flash-attention仅-nvidia)）；
- **3.11 也能用**（`torch==2.7.0` 支持 3.9–3.13），但**拿不到预编译的 flash-attention**，只能跳过或自行从源码编译；
- **避开 3.12+**：PyTorch 2.7 虽然支持，但 `qwen-tts` / `transformers` 组合未经本项目验证。

```bash
# 查看可用的 3.10 补丁版本
pyenv install --list | grep -E '^ *3\.10\.'

# 安装最新 3.10.x（下行版本号需替换为上面列表中的最新值）
pyenv install 3.10.16
```

> Linux 上这是**源码编译**，耗时 3–10 分钟，属正常现象。

### 4.2 创建虚拟环境

**方式 A：pyenv-virtualenv（推荐，进目录自动激活）**

```bash
cd /path/to/alexandria-audiobook

pyenv virtualenv 3.10.16 alexandria
pyenv local alexandria          # 在仓库根写入 .python-version
python -V                       # 应输出 Python 3.10.16
```

之后**每次 `cd` 进仓库都会自动激活**，`pip install` 一律装进该虚拟环境。

> `.python-version` 是点号开头的文件，本仓库 `.gitignore` 的 `.*` 规则已忽略它，不会污染 `git status`。

**方式 B：pyenv + 标准库 venv**

```bash
cd /path/to/alexandria-audiobook
pyenv local 3.10.16
python -m venv app/env          # 与 Pinokio 的目录布局一致
source app/env/bin/activate
```

> 用 `app/env` 有个额外好处：`pinokio.js` 通过 `info.exists("app/env")` 判断是否"已安装"，所以这个环境**同时也能被 Pinokio 识别为已安装**，两套运行方式可以共用。

---

## 5. 安装依赖

```bash
cd /path/to/alexandria-audiobook
python -V                       # 确认在虚拟环境里
python -m pip install --upgrade pip setuptools wheel
```

### 5.1 Python 包（顺序与 `install.js` 一致）

```bash
# 1) 应用依赖
pip install -r app/requirements.txt

# 2) TTS 引擎（不在 requirements.txt 中，必须单独装）
pip install qwen-tts==0.1.1

# 3) PyTorch —— 最后装，见 5.2
```

> `app/requirements.txt` 中**不含 torch**，它由平台的 wheel 索引决定，必须单独安装。
> `install.js` 还会先执行一次 `uv pip uninstall google-genai` 来避免依赖冲突；如果你是全新环境可以忽略。

### 5.2 安装 PyTorch（按平台二选一）

与 `torch.js` 一致：**最后安装，`--force-reinstall --no-deps` 覆盖掉 `qwen-tts` 可能拉进来的版本**。

**NVIDIA GPU（CUDA 12.8，需驱动 550+）**

```bash
# Linux
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps

# Windows（命令相同）
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

**AMD GPU（Linux + ROCm）**

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall --no-deps
pip install pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm6.3 || true
```

> 把 `rocm6.3` 换成你本机的 ROCm 版本（`cat /opt/rocm/.info/version`）。ROCm 7.x 用 `rocm7.2`。

**macOS（Apple Silicon / Intel）**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

**纯 CPU（任何平台）**

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
```

### 5.3 可选：flash-attention（仅 NVIDIA）

`install.js` 会传 `flashattention: true`，为 NVIDIA 用户安装预编译 wheel 以加速编码器。**这是可选优化**，跳过不影响功能（唯一代价是编码稍慢，以及官方 README 提到的"faster encoding"特性）。

> ⚠️ **这两个 wheel 是 `cp310`，只能在 Python 3.10 上安装。** 用 3.11 的话 pip 会直接报 `is not a supported wheel on this platform` —— 跳过本节即可，或自行[从源码编译](https://github.com/Dao-AILab/flash-attention)。

```bash
# Python 3.10 + Linux x86_64
pip install https://huggingface.co/cocktailpeanut/wheels/resolve/main/flash_attn-2.8.3%2Bcu128torch2.7-cp310-cp310-linux_x86_64.whl

# Python 3.10 + Windows
pip install https://huggingface.co/cocktailpeanut/wheels/resolve/main/flash_attn-2.8.2%2Bcu128torch2.7-cp310-cp310-win_amd64.whl
```

### 5.4 验证安装

```bash
python -c "import torch, fastapi, uvicorn, soundfile, pydub; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

---

## 6. 启动

```bash
cd /path/to/alexandria-audiobook
python app/app.py
```

看到以下输出即成功：

```
INFO:     Started server process [...]
INFO:     Uvicorn running on http://127.0.0.1:4200
```

浏览器打开 **<http://127.0.0.1:4200>**（默认端口 4200）。

> 也可以 `cd app && python app.py` —— `start.js` 就是这么做的。两种方式等价：应用内部所有子进程都显式使用 `app/` 作为工作目录，`default_prompts.txt` 等文件也是相对 `app/` 的父目录解析的，所以**从哪个目录启动都行**。

**停止**：在终端按 `Ctrl+C`。

### 6.1 做成快捷命令（可选）

```bash
cd /path/to/alexandria-audiobook
cat > start.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python app/app.py
EOF
chmod +x start.sh
./start.sh
```

> `start.sh` 不在 `.gitignore` 里，会让 `git status` 多出一个未跟踪文件。不想看到它就加进本地排除：
>
> ```bash
> echo 'start.sh' >> .git/info/exclude
> ```

---

## 7. 环境变量

全部有默认值，不设置即可运行：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `ALEXANDRIA_HOST` | `127.0.0.1` | 监听地址。设 `0.0.0.0` 可让局域网/容器访问 |
| `ALEXANDRIA_PORT` | `4200` | 监听端口。端口被占用时改这里 |
| `ALEXANDRIA_CONFIG_PATH` | `app/config.json` | WebUI 配置文件的存放路径 |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 模型缓存位置（TTS 权重落在这里） |

示例：

```bash
ALEXANDRIA_PORT=8080 ALEXANDRIA_HOST=0.0.0.0 python app/app.py
```

---

## 8. 首次运行会发生什么

1. **必须先在 Setup 标签页配好 LLM**（Base URL / API Key / 模型名），点 **Save Configuration**。
2. **首次生成语音会下载 ~3.5 GB 模型**，这是正常的：
   - 每个模型变体（CustomVoice / Base-Clone / VoiceDesign）各约 3.5 GB；
   - 只有你实际用到的变体才会下载（多数人从 CustomVoice 开始）；
   - 下载时 Web UI 看起来像卡住，实际是在等下载完成；
   - 权重缓存在 `HF_HOME`，之后加载只需几秒。
3. **第一个批次额外慢**：MIOpen 自动调优（AMD）或 codec 编译（若开启）会有一次性 30–60 秒预热。
4. **显存决定并发量**：显存不足就在 Setup 里调小 **Parallel Workers** 或 **Max Chars/Batch**。

> 详细日志直接打印在运行 `python app/app.py` 的终端里 —— 这在 Pinokio 里是"Terminal"面板，手动运行时就是你的终端窗口。

---

## 9. 模型下载策略（external 模式）

Alexandria 的 TTS 有两种模式：

| 模式 | 行为 |
|---|---|
| `local` | 内置 Qwen3-TTS，直接加载本地权重（会下载 ~3.5 GB/变体） |
| `external` | 连接到远程 Qwen3-TTS Gradio 服务器，**禁止下载任何模型** |

**当 TTS 模式设为 `external` 时，本仓库会拒绝一切模型下载**，避免在只做 UI 的机器上意外拉取数 GB 权重：

- `TTSEngine` 的模型加载永远不会回退到 HuggingFace 仓库 ID —— 权重不在本地缓存时直接抛错；
- 内置 LoRA 音色的自动下载被禁用；
- 相关接口返回 **403**，并在错误信息里提示切换模式；
- LoRA 训练子进程会带上 `--no-download`，只复用缓存，不再拉取 Base 模型；
- **已经下载好的缓存权重仍然可以正常复用**，只是不会再发起新的下载。

这条策略让 external 模式成为"零下载"的理想选择：在 macOS 这类只能跑 CPU 的机器上，把 Web UI 留在本地、把 TTS 卸载到远程 GPU 服务器。

---

## 10. 更新与重置

### 10.1 更新（等价于 `update.js`）

```bash
cd /path/to/alexandria-audiobook
git pull
pip install -r app/requirements.txt
pip install qwen-tts==0.1.1
# 依赖大版本变化时再重装 torch，见 5.2
```

### 10.2 重置（等价于 `reset.js`）

删除生成产物，保留代码与模型缓存：

```bash
cd /path/to/alexandria-audiobook
rm -rf voicelines
rm -f annotated_script.json voices.json voice_config.json state.json \
      chunks.json cloned_audiobook.mp3 app/config.json
```

> 这会清空脚本、音色配置和已生成音频。**模型权重在 `HF_HOME` 里，不受影响**，无需重新下载。
>
> 如果想连模型一起清掉：`rm -rf ~/.cache/huggingface`（或你的 `HF_HOME`）。

---

## 11. 故障排查

| 现象 | 原因与解决 |
|---|---|
| `Refusing to download '...'` | TTS 模式是 `external`，且权重不在本地缓存。切到 `local` 模式，或先在同一台机器上用 `local` 模式跑一次把权重缓存下来 |
| `ModuleNotFoundError: No module named 'qwen_tts'` | 漏装了第 5.1 步的第 2 条：`pip install qwen-tts==0.1.1` |
| `torch.cuda.is_available()` 返回 `False` | torch 装成了 CPU 版。按[第 5.2 节](#52-安装-pytorch按平台二选一)用对应索引重装（注意 `--force-reinstall`） |
| MP3 导出变成 WAV | ffmpeg 缺失或缺少 `libmp3lame`。`brew install ffmpeg` / 换用带完整编码器的构建 |
| M4B 导出报错 | `app/project.py` 直接调用 `ffmpeg` 的 `aac` 编码器，确认 ffmpeg 在 PATH 中 |
| macOS 上极慢 | 预期行为 —— Apple Silicon 不支持 MPS。建议在 `app/config.json` 里显式设 `"tts": { "device": "cpu" }`（`tts.py` 的 `auto` 探测会返回 `mps`，与官方说明不一致），或改用 external 模式 |
| 端口被占用 | `ALEXANDRIA_PORT=8080 python app/app.py` |
| `pyenv install` 编译失败 | 缺[第 3.2 节](#32-编译依赖)的编译依赖；查看 `~/.pyenv/logs/` 下的构建日志 |
| `python -V` 不是 3.10 | 虚拟环境未激活。`pyenv local alexandria` 后重新 `cd` 进目录，或手动 `pyenv activate alexandria` |
| `flash_attn-...+cp310-...whl is not a supported wheel on this platform` | 你用 Python 3.11（或更高）装 3.10 专用的 flash-attention 轮子。跳过这一可选步骤，或改用 Python 3.10 |
| Preparer 标签页返回 503 | 仓库不含 `app/alexandria_preparer.py`，该功能在开源版本中不可用（与是否用 Pinokio 无关） |

更多常见问题见上游 [Wiki / Troubleshooting](https://github.com/Finrandojin/alexandria-audiobook/wiki/Troubleshooting)。

---

## 12. 可选方案：Docker / Colab

如果不想在本机编译 Python、装 CUDA 轮子，仓库还提供两条官方路径：

**Docker（需要 NVIDIA GPU + NVIDIA Container Toolkit）**

```bash
git clone https://github.com/Finrandojin/alexandria-audiobook.git
cd alexandria-audiobook
docker compose up --build     # http://localhost:4200
```

用户数据与模型缓存在 `./data/` 和 Docker volume 中持久化。

**Google Colab（无需本地 GPU）**

打开仓库内的 `alexandria_colab.ipynb`，在免费 T4 上运行，通过 Colab 内置端口转发访问 Web UI。

---

## 附：一页速查

```bash
# --- 一次性准备 ---
brew install pyenv pyenv-virtualenv ffmpeg          # macOS
# 或 curl -fsSL https://pyenv.run | bash            # Linux
pyenv install 3.10.16

# --- 每个仓库 ---
cd /path/to/alexandria-audiobook
pyenv virtualenv 3.10.16 alexandria
pyenv local alexandria

pip install -r app/requirements.txt
pip install qwen-tts==0.1.1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps   # 按平台调整

# --- 运行 ---
python app/app.py        # → http://127.0.0.1:4200
```
