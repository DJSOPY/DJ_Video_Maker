#!/bin/bash
# ============================================================
#  🎧 DJ Video Maker — サーバー起動（ブラウザで操作する版）
#  ダブルクリックすると準備をして、ブラウザが自動で開きます。
#  ブラウザ上で 曲をドロップ → 作成 → ダウンロード ができます。
#  ※このウィンドウは開いたままにしてください（閉じると停止します）。
#  ※setup_common.sh が同じフォルダに必要です
# ============================================================
cd "$(dirname "$0")"
clear
echo "========================================================"
echo "  🎧 DJ Video Maker（ブラウザ版）を準備しています..."
echo "========================================================"
echo ""

# ---- 共通セットアップ部品を読み込む ----
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$DIR/setup_common.sh" ]; then
    echo "❌ setup_common.sh が見つかりません（この.commandと同じフォルダに置いてください）"
    read -p "Enterで閉じる..."; exit 1
fi
source "$DIR/setup_common.sh"

# ---- 診断 → ツール → Python → 検証（AIライブラリは静かに試すだけ）----
djvm_full_setup lite

# ---- 必要ファイル確認 ----
for f in dj_maker_core.py web_server.py web_ui.html; do
    [ -f "$DIR/$f" ] || { echo "❌ $f が同じフォルダにありません"; read -p "Enterで閉じる..."; exit 1; }
done
find "$DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# ---- サーバー起動＋ブラウザを開く ----
export DJVM_PYTHON="$PYTHON_CMD"
# 前回のサーバーが残っていたら止める（ポート衝突を防ぐ）
pkill -f "web_server.py" 2>/dev/null; sleep 1
PORTFILE="$HOME/.dj_video_maker/web_port"
rm -f "$PORTFILE" 2>/dev/null
echo ""
echo "✅ 準備完了！サーバーを起動します（ブラウザが自動で開きます）"
# サーバーが実際に使うポートを web_port に書くので、それを読んでブラウザを開く
(
  for i in $(seq 1 20); do
    [ -f "$PORTFILE" ] && { open "http://127.0.0.1:$(cat "$PORTFILE")"; break; }
    sleep 0.5
  done
) &
"$PYTHON_CMD" "$DIR/web_server.py"
