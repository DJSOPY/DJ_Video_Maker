#!/bin/bash
# ============================================================
#  🎧 DJ Video Maker — 誰のMacでも動く汎用版
#  初回はセットアップを自動で行います（数分かかります）
# ============================================================
cd "$(dirname "$0")"
# ---- Gatekeeper対策：配布物についた隔離属性を自分で外す ----
# ネット経由で受け取ったDMG/zipの中身には com.apple.quarantine が付き、
# 「開発元が未確認のため開けません」になる。この.commandが一度動いた時点で
# 同じフォルダの .py / .command から隔離属性を落としておく（次回以降は無警告）。
xattr -dr com.apple.quarantine "$(pwd)" 2>/dev/null
clear
echo "========================================================"
echo "  🎧 DJ Video Maker  —  Serato / VirtualDJ 対応"
echo "        created by DJ SOPY / @sousouagain"
echo "========================================================"
echo ""

# ---- Homebrew のPATHを通す（Apple Silicon / Intel 両対応）----
[ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
[ -x /usr/local/bin/brew ] && eval "$(/usr/local/bin/brew shellenv)"

# ---- Homebrew ----
if ! command -v brew &>/dev/null; then
    # 先に「管理者かどうか」を確認（管理者でないとHomebrewは入れられない＝macOSの仕様）
    if ! groups | grep -qw admin; then
        echo ""
        echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
        echo "┃ ❌ このMacのアカウント（$(whoami)）は【管理者】ではないため、  "
        echo "┃    初回セットアップができません（macOSの仕様です）。          "
        echo "┃                                                              "
        echo "┃ 【直し方】※データは一切消えません                            "
        echo "┃  1. システム設定 → ユーザとグループ → このアカウントの(i)     "
        echo "┃     →「このユーザを管理者として設定」をON → Macを再起動      "
        echo "┃  2. もう一度このアイコンをダブルクリック                      "
        echo "┃                                                              "
        echo "┃  詳しくは同梱の説明書（README）の🔑の章を見てください。       "
        echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        echo ""
        read -p "Enterで閉じる..."; exit 1
    fi
    echo "📦 初回準備：基本ツール(Homebrew)を入れています..."
    echo "   ※Macのパスワードを1回だけ聞かれます（macOSの仕様）。入力して進めてください。"
    # 確認(Press RETURN)を出さず自動で進める
    export NONINTERACTIVE=1
    export HOMEBREW_NO_ENV_HINTS=1
    export HOMEBREW_NO_ANALYTICS=1
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" < /dev/null
    [ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
    [ -x /usr/local/bin/brew ] && eval "$(/usr/local/bin/brew shellenv)"
    if ! command -v brew &>/dev/null; then
        echo "❌ Homebrewのインストールに失敗しました"
        echo "   ・ネット接続を確認してください"
        echo "   ・パスワードの入力を求められた場合は、Macのパスワードを打ってEnter（画面には表示されません）"
        read -p "Enterで閉じる..."; exit 1
    fi
fi
export HOMEBREW_NO_ENV_HINTS=1
export HOMEBREW_NO_AUTO_UPDATE=1

# ---- ffmpeg（映像処理の要。ffprobe もこれに含まれる）----
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    echo "📦 ffmpeg をインストール中..."
    brew install ffmpeg
    # 失敗した場合、古い壊れた依存(openssl@1.1 等)の残骸が原因のことが多いので自己修復して再試行
    if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
        echo "🩹 うまく入らなかったため、壊れた古い依存を掃除して再試行します..."
        brew uninstall --ignore-dependencies openssl@1.1 2>/dev/null
        brew cleanup 2>/dev/null
        brew update 2>/dev/null
        brew install ffmpeg
    fi
    if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
        echo ""
        echo "❌ ffmpeg（映像処理ツール）のインストールに失敗しました。"
        echo "   お手数ですが、ターミナルで次の2行を手で実行してから、もう一度お試しください："
        echo "       brew doctor"
        echo "       brew install ffmpeg"
        read -p "Enterで閉じる..."; exit 1
    fi
fi

# ---- yt-dlp ----
if ! command -v yt-dlp &>/dev/null; then
    echo "📦 yt-dlp をインストール中..."
    brew install yt-dlp
    if ! command -v yt-dlp &>/dev/null; then
        echo "🩹 うまく入らなかったため、壊れた古い依存を掃除して再試行します..."
        brew uninstall --ignore-dependencies openssl@1.1 2>/dev/null
        brew cleanup 2>/dev/null
        brew install yt-dlp
    fi
    if ! command -v yt-dlp &>/dev/null; then
        echo ""
        echo "❌ yt-dlp（動画ダウンロードツール）のインストールに失敗しました。"
        echo "   ターミナルで次を手で実行してから、もう一度お試しください："
        echo "       brew install yt-dlp"
        read -p "Enterで閉じる..."; exit 1
    fi
else
    # YouTubeの仕様変更対策: 週1回だけ自動アップデート
    STAMP="$HOME/.dj_video_maker/ytdlp_updated"
    mkdir -p "$HOME/.dj_video_maker"
    if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(stat -f %m "$STAMP" 2>/dev/null || echo 0) )) -gt 604800 ]; then
        echo "🔄 yt-dlp を最新版に更新中（週1回の自動チェック）..."
        brew upgrade yt-dlp 2>/dev/null
        touch "$STAMP"
    fi
fi

# ---- Python venv（ユーザーごとのホームに作成）----
VENV="$HOME/.dj_video_maker_env"
PYTHON_CMD="$VENV/bin/python3"

if [ ! -f "$PYTHON_CMD" ]; then
    echo "📦 Python環境を構築中..."
    # brewのpython3を優先、なければシステムpython3
    BREW_PY="$(brew --prefix 2>/dev/null)/bin/python3"
    if [ -f "$BREW_PY" ]; then
        "$BREW_PY" -m venv "$VENV"
    else
        python3 -m venv "$VENV"
    fi
    if [ ! -f "$PYTHON_CMD" ]; then
        echo "❌ Python環境の作成に失敗しました"
        read -p "Enterで閉じる..."; exit 1
    fi
fi

# ---- numpy / scipy / Pillow ----
if ! "$PYTHON_CMD" -c "import numpy, scipy, mutagen, PIL, librosa, fastdtw" &>/dev/null; then
    echo "📦 基本ライブラリをインストール中（初回・数分）..."
    "$PYTHON_CMD" -m pip install --quiet --no-input --upgrade pip
    "$PYTHON_CMD" -m pip install --quiet --no-input numpy scipy mutagen Pillow librosa fastdtw \
      || "$PYTHON_CMD" -m pip install --no-input numpy scipy mutagen Pillow librosa fastdtw
    if ! "$PYTHON_CMD" -c "import numpy, scipy, mutagen" &>/dev/null; then
        echo "❌ 基本ライブラリのインストールに失敗しました（ネット接続を確認してください）"
        read -p "Enterで閉じる..."; exit 1
    fi
fi

# ---- Pro版（高精度リップシンク）用ライブラリ（任意・無くても従来方式で動く）----
if ! "$PYTHON_CMD" -c "import torch, demucs" &>/dev/null; then
    echo ""
    echo "🧠 高精度リップシンク用のAIを準備しています。"
    echo "   数GBのダウンロードがあり、回線によっては 10〜30分 かかります。"
    echo "   ☕️ 進捗バーが動いている間はそのままお待ちください（初回だけ・閉じないで）"
    echo ""
    # --quietを付けない＝進捗が見えて「固まった？」を防ぐ。失敗したら1回だけリトライ
    "$PYTHON_CMD" -m pip install --no-input demucs \
      || "$PYTHON_CMD" -m pip install --no-input demucs \
      || echo "   （高精度ライブラリは入りませんでした → 従来方式で動きます）"
fi
# HuBERT用（transformers / torchaudio）。★バージョン固定＝毎回の再取得を防ぐ。
# 固定しないと transformers が起動のたびに tokenizers を別版へ入れ替え、
# 次回また不整合→再インストール…と、波形一致の曲でも起動が重くなる。
# tokenizers を transformers 側の要求域にピン留めしておく（faster-whisper と両立）。
if ! "$PYTHON_CMD" -c "import transformers, torchaudio" &>/dev/null; then
    "$PYTHON_CMD" -m pip install --no-input "numpy<2" "transformers==4.44.2" "tokenizers>=0.19,<0.20" torchaudio \
      || "$PYTHON_CMD" -m pip install --no-input "numpy<2" "transformers==4.44.2" "tokenizers>=0.19,<0.20" torchaudio \
      || echo "   （HuBERT用ライブラリは入りませんでした → MFCC/従来方式で動きます）"
fi
# ---- torchcodec（torchaudio 2.9以降は音声の書き出しにこれが必須。無いとDemucsが保存で落ちる）----
# 本体は torchcodec 無しでも動く（vocal_sync が demucs API で保存経路を回避する）が、
# 入っていればより多くの経路が使えるので、静かに入れておく。
if "$PYTHON_CMD" -c "import torchaudio" &>/dev/null && ! "$PYTHON_CMD" -c "import torchcodec" &>/dev/null; then
    "$PYTHON_CMD" -m pip install --quiet --no-input torchcodec 2>/dev/null \
      || echo "   （torchcodecは入りませんでした → 保存経路を使わない方式で動きます）"
fi
# ---- Forced Alignment（Whisper単語タイムスタンプ＝口パク精度の最終段。任意）----
if ! "$PYTHON_CMD" -c "import faster_whisper" &>/dev/null; then
    "$PYTHON_CMD" -m pip install --no-input faster-whisper \
      || "$PYTHON_CMD" -m pip install --no-input faster-whisper \
      || echo "   （Whisper単語アライメントは入りませんでした → その段はスキップして動きます）"
fi
# ---- WhisperX（任意）----
# Python 3.14 では現行WhisperX系の依存が未対応で、毎回pipが長く探した末に失敗しやすい。
# 通常は faster-whisper で十分動くため、明示指定時だけ試す。
if [ "${DJVM_INSTALL_WHISPERX:-0}" = "1" ]; then
    if ! "$PYTHON_CMD" -c "import whisperx" &>/dev/null; then
        "$PYTHON_CMD" -m pip install --no-input whisperx \
          || echo "   （WhisperXは入りませんでした → faster-whisperで単語アライメントします）"
    fi
fi
# ---- 口の動き解析（mouth_sync：歌ってない区間の口パク映像を回避・後半ズレ補正。任意）----
# mediapipe/opencv が無くても本体は従来通り動く（mouth_sync は自動でスキップ）。
if ! "$PYTHON_CMD" -c "import mediapipe, cv2" &>/dev/null; then
    "$PYTHON_CMD" -m pip install --no-input mediapipe opencv-python \
      || "$PYTHON_CMD" -m pip install --no-input mediapipe opencv-python \
      || echo "   （口の動き解析ライブラリは入りませんでした → その機能はスキップして動きます）"
fi
echo ""
# ---- 起動前の最終チェック：必須ツールが本当に使えるか確認（入ってないのに『準備完了』と言わない）----
_MISSING=""
for _t in ffmpeg ffprobe yt-dlp; do
    command -v "$_t" &>/dev/null || _MISSING="$_MISSING $_t"
done
"$PYTHON_CMD" -c "import numpy, scipy, librosa" 2>/dev/null || _MISSING="$_MISSING pythonライブラリ"
if [ -n "$_MISSING" ]; then
    echo ""
    echo "❌ セットアップが完了していません。次が使えません：$_MISSING"
    echo "   ネット接続を確認して、もう一度このアイコンをダブルクリックしてください。"
    echo "   何度も失敗する場合は、ターミナルで  brew doctor  を実行して指示に従ってください。"
    read -p "Enterで閉じる..."; exit 1
fi

echo "✅ 準備完了！ツールを起動します..."

# ---- 必要ファイルの確認（全部入りには 3つの.py が同じフォルダに要る）----
DIR="$(dirname "$0")"
[ -f "$DIR/dj_maker_core.py" ] || { echo "❌ dj_maker_core.py が見つかりません（この.commandと同じフォルダに置いてください）"; read -p "Enterで閉じる..."; exit 1; }
if [ ! -f "$DIR/vocal_sync.py" ] || [ ! -f "$DIR/lipsync_pro.py" ]; then
    echo "⚠️ 同じフォルダに lipsync_pro.py / vocal_sync.py が揃っていません。"
    [ -f "$DIR/lipsync_pro.py" ] || echo "    → lipsync_pro.py が無い＝高精度リップシンクは使えず従来方式になります"
    [ -f "$DIR/vocal_sync.py" ]  || echo "    → vocal_sync.py が無い＝リップシンク自体が動きません"
    echo "    この .command と同じフォルダに、dj_maker_core.py / vocal_sync.py / lipsync_pro.py を全部置いてください。"
    echo ""
fi
[ -f "$DIR/mouth_sync.py" ] || echo "ℹ️ mouth_sync.py が無い＝口の動き解析はスキップ（本体は従来通り動きます）"

# ---- 古いキャッシュを毎回自動削除（.pyを差し替えても確実に新しいコードで動く）----
# __pycache__/*.pyc が残っていると、ソースを更新しても古い挙動が出ることがある。
# 起動のたびにこのフォルダのキャッシュを消すことで、それを根本的に防ぐ。
find "$DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "$DIR" -name "*.pyc" -delete 2>/dev/null

# ---- 起動 ----
exec "$PYTHON_CMD" "$DIR/dj_maker_core.py"
