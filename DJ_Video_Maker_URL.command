#!/bin/bash
# ============================================================
#  🎧 DJ Video Maker (URL版) — MVのYouTube URLを自分で貼るモード
#  初回はセットアップを自動で行います（数分かかります）
#  ※setup_common.sh が同じフォルダに必要です
# ============================================================
cd "$(dirname "$0")"
clear
echo "========================================================"
echo "  🎧 DJ Video Maker (URL版)  —  Serato / VirtualDJ 対応"
echo "        created by DJ SOPY / @sousouagain"
echo "========================================================"
echo ""

# ---- 共通セットアップ部品を読み込む ----
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$DIR/setup_common.sh" ]; then
    echo "❌ setup_common.sh が見つかりません（この.commandと同じフォルダに置いてください）"
    read -p "Enterで閉じる..."; exit 1
fi
source "$DIR/setup_common.sh"

# ---- 診断 → ツール → Python → 検証（すべて共通部品が実施）----
djvm_full_setup full

echo ""
echo "✅ 準備完了！ツールを起動します..."

# ---- 必要ファイルの確認（全部入りには 3つの.py が同じフォルダに要る）----
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
find "$DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "$DIR" -name "*.pyc" -delete 2>/dev/null

# ---- 起動（URLモード：自動検索せず、曲ごとにYouTube URLを貼ってもらう）----
export DJVM_MANUAL_URL=1
exec "$PYTHON_CMD" "$DIR/dj_maker_core.py"
