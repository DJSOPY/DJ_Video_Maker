#!/bin/bash
# ============================================================
#  🛠 DJ Video Maker — 修復（初回からやり直し）
#  うまく動かない / 途中で落ちる ときに、これをダブルクリック。
#  壊れた古い残骸を掃除して、必要なものを一から入れ直します。
#  ※音楽・写真・書類などの個人データには一切触りません。
# ============================================================
cd "$(dirname "$0")"
clear
echo "========================================================"
echo "  🛠 DJ Video Maker 修復ツール"
echo "     壊れた残骸を掃除して、必要なものを入れ直します"
echo "========================================================"
echo ""

# ---- Homebrew のPATH（Apple Silicon / Intel 両対応）----
[ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
[ -x /usr/local/bin/brew ]    && eval "$(/usr/local/bin/brew shellenv)"

# 管理者チェック（Homebrew操作に必要）
if ! command -v brew &>/dev/null && ! groups | grep -qw admin; then
    echo "❌ このアカウント（$(whoami)）は管理者ではないため修復できません。"
    echo "   システム設定 → ユーザとグループ で管理者にして再起動してから、もう一度実行してください。"
    read -p "Enterで閉じる..."; exit 1
fi

# ---- ① 壊れた残骸の掃除 ----
echo "🧹 [1/5] 壊れた古い残骸を掃除しています..."
# openssl@1.1 の壊れたリンク（is not a valid keg の原因）を直接削除
rm -f /usr/local/opt/openssl@1.1 /opt/homebrew/opt/openssl@1.1 2>/dev/null
brew uninstall --ignore-dependencies openssl@1.1 2>/dev/null
# Pythonキャッシュ・pyc の死骸
find "$(dirname "$0")" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "$(dirname "$0")" -name "*.pyc" -delete 2>/dev/null
if command -v brew &>/dev/null; then
    brew cleanup 2>/dev/null
fi
echo "   完了"

# ---- ② Homebrew（無ければ入れる / あれば更新）----
echo "🍺 [2/5] 基本ツール(Homebrew)を確認・更新しています..."
if ! command -v brew &>/dev/null; then
    echo "   Homebrewが無いので入れます（Macのパスワードを1回聞かれます）..."
    export NONINTERACTIVE=1 HOMEBREW_NO_ENV_HINTS=1 HOMEBREW_NO_ANALYTICS=1
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" < /dev/null
    [ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
    [ -x /usr/local/bin/brew ]    && eval "$(/usr/local/bin/brew shellenv)"
fi
if ! command -v brew &>/dev/null; then
    echo "❌ Homebrewが使えません。ネット接続と管理者権限を確認してください。"
    read -p "Enterで閉じる..."; exit 1
fi
export HOMEBREW_NO_ENV_HINTS=1
brew update 2>/dev/null

# ---- ③ ffmpeg / yt-dlp を入れ直す ----
echo "🎬 [3/5] ffmpeg と yt-dlp を入れ直しています..."
brew reinstall ffmpeg 2>/dev/null || brew install ffmpeg
brew reinstall yt-dlp 2>/dev/null || brew install yt-dlp
brew upgrade yt-dlp 2>/dev/null

# ---- ④ Python環境を作り直す ----
echo "🐍 [4/5] Python環境とライブラリを入れ直しています（数分かかります）..."
VENV="$HOME/.dj_video_maker_env"
PYTHON_CMD="$VENV/bin/python3"
# venvを作り直す（壊れている可能性があるので一度消す）
rm -rf "$VENV" 2>/dev/null
# venvの作成・pip更新・全ライブラリ導入は setup_common.sh に一本化する。
# ★ライブラリの導入は setup_common.sh の検証済みロジックに一本化する。
#   （numpy/scipy/mediapipe/transformers のバージョン固定・jax回避・衝突対策は
#     すべてそこに集約。ここに個別 pip を書くと設定がズレて過去のバグが再発する。）
SETUP_SH="$(dirname "$0")/setup_common.sh"
if [ -f "$SETUP_SH" ]; then
    export PYTHON_CMD
    # shellcheck disable=SC1090
    . "$SETUP_SH"
    djvm_setup_python full
else
    echo "❌ setup_common.sh が見つかりません（フォルダの中身が欠けています）。"
    echo "   install.sh からダウンロードし直してください。"
    read -p "Enterで閉じる..."; exit 1
fi

# ---- ⑤ 動作検証 ----
echo "🔍 [5/5] 正しく入ったか確認しています..."
NG=""
command -v ffmpeg  &>/dev/null || NG="$NG ffmpeg"
command -v ffprobe &>/dev/null || NG="$NG ffprobe"
command -v yt-dlp  &>/dev/null || NG="$NG yt-dlp"
"$PYTHON_CMD" -c "import numpy, scipy, librosa, mutagen" 2>/dev/null || NG="$NG pythonライブラリ"

echo ""
echo "========================================================"
if [ -z "$NG" ]; then
    echo "  ✅ 修復完了！すべて正常に入りました。"
    echo "     ffmpeg : $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
    echo "     yt-dlp : $(yt-dlp --version 2>/dev/null)"
    echo ""
    echo "  → 通常の DJ_Video_Maker.command（または URL版）を"
    echo "     ダブルクリックして使ってください。"
else
    echo "  ⚠️ まだ次が入っていません：$NG"
    echo ""
    echo "  ネット接続を確認して、もう一度この修復ツールを実行してください。"
    echo "  それでもダメな場合は、ターミナルで次を実行して指示に従ってください："
    echo "      brew doctor"
fi
echo "========================================================"
echo ""
read -p "Enterで閉じる..."
