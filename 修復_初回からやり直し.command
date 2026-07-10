#!/bin/bash
# ============================================================
#  🛠 DJ Video Maker — 修復（初回からやり直し）
#  うまく動かない / 途中で落ちる ときに、これをダブルクリック。
#  壊れた古い残骸を掃除して、必要なものを一から入れ直します。
#  Homebrewが入れられない環境では、パスワード不要の
#  直接ダウンロード方式に自動で切り替わります。
#  ※音楽・写真・書類などの個人データには一切触りません。
#  ※setup_common.sh が同じフォルダに必要です
# ============================================================
cd "$(dirname "$0")"
clear
echo "========================================================"
echo "  🛠 DJ Video Maker 修復ツール"
echo "     壊れた残骸を掃除して、必要なものを入れ直します"
echo "========================================================"
echo ""

# ---- 共通セットアップ部品を読み込む ----
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$DIR/setup_common.sh" ]; then
    echo "❌ setup_common.sh が見つかりません（この.commandと同じフォルダに置いてください）"
    read -p "Enterで閉じる..."; exit 1
fi
source "$DIR/setup_common.sh"

# ---- 事前診断（ネット・開発ツール）----
djvm_check_network
djvm_check_clt

# ---- ① 壊れた残骸の掃除 ----
echo "🧹 [1/5] 壊れた古い残骸を掃除しています..."
# openssl@1.1 の壊れたリンク（is not a valid keg の原因）を直接削除
rm -f /usr/local/opt/openssl@1.1 /opt/homebrew/opt/openssl@1.1 2>/dev/null
command -v brew &>/dev/null && brew uninstall --ignore-dependencies openssl@1.1 2>/dev/null
# Pythonキャッシュ・pyc の死骸
find "$DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "$DIR" -name "*.pyc" -delete 2>/dev/null
# 以前の静的バイナリも一旦消して入れ直す（半端なDLの残骸対策）
rm -rf "$DJVM_BIN" 2>/dev/null
command -v brew &>/dev/null && brew cleanup 2>/dev/null
echo "   完了"

# ---- ② ツールを入れ直す（brew優先 → だめなら直接ダウンロード）----
echo "🎬 [2/5] ffmpeg / ffprobe / yt-dlp を入れ直しています..."
if command -v brew &>/dev/null; then
    brew update 2>/dev/null
    brew reinstall ffmpeg 2>/dev/null || brew install ffmpeg
    brew reinstall yt-dlp 2>/dev/null || brew install yt-dlp
    brew upgrade yt-dlp 2>/dev/null
fi
# brewが無い/失敗した分は共通部品が面倒を見る（Homebrew導入 or 静的DL）
djvm_ensure_tools

# ---- ③ Python環境を作り直す ----
echo "🐍 [3/5] Python環境を作り直しています..."
rm -rf "$HOME/.dj_video_maker_env" 2>/dev/null

# ---- ④ ライブラリを入れ直す ----
echo "📚 [4/5] ライブラリを入れ直しています（数分かかります）..."
djvm_setup_python full

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
    echo "  → 『これをダブルクリック！.command』（または DJ_Video_Maker.command）を"
    echo "     ダブルクリックして使ってください。"
else
    echo "  ⚠️ まだ次が入っていません：$NG"
    echo ""
    echo "  ・回線を変えて（VPNオフ／iPhoneテザリング等）もう一度実行してください。"
    echo "  ・それでもダメな場合は、この画面を写真に撮って配布元に送ってください。"
fi
echo "========================================================"
echo ""
read -p "Enterで閉じる..."
