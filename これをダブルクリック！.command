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

# ============================================================
#  🔄 自動アップデート（1日1回・安全策込み）
#   GitHubの最新版を静かにチェックし、更新があればダウンロードして差し替える。
#   - 1日1回だけチェック（前回から24時間未満ならスキップ＝普段は一瞬）
#   - 一時フォルダに全部落として、成功した分だけ差し替える（壊れない）
#   - ネットが無い・失敗しても、今あるファイルのまま必ず起動する
#   - この.command自身は書き換えない（実行中ファイル書き換えを避ける）
# ============================================================
djvm_auto_update() {
    local DIR="$1"
    local BASE="https://raw.githubusercontent.com/DJSOPY/DJ_Video_Maker/main"
    local STAMP="$HOME/.dj_video_maker/last_update_check"
    mkdir -p "$HOME/.dj_video_maker" 2>/dev/null

    # 1日1回だけ：前回チェックから24時間(86400秒)未満なら何もしない
    if [ -f "$STAMP" ]; then
        local last now
        last=$(cat "$STAMP" 2>/dev/null || echo 0)
        now=$(date +%s)
        if [ $((now - last)) -lt 86400 ]; then
            return 0
        fi
    fi

    # オフラインなら静かにスキップ（起動は止めない）
    if ! curl -m 8 -s -o /dev/null -I "https://raw.githubusercontent.com" 2>/dev/null; then
        return 0
    fi

    echo "🔄 最新版があるか確認しています..."
    local TMP; TMP="$(mktemp -d)"
    # 更新対象＝本体ファイルのみ（.command自身は含めない）
    local FILES="dj_maker_core.py lipsync_pro.py vocal_sync.py mouth_sync.py web_server.py web_ui.html setup_common.sh"
    local updated=0 failed=0

    for f in $FILES; do
        # 既存と同じ内容ならスキップ（無駄な差し替えをしない）
        if curl -fsSL -m 30 "$BASE/$f" -o "$TMP/$f" 2>/dev/null; then
            if [ -f "$DIR/$f" ] && cmp -s "$TMP/$f" "$DIR/$f"; then
                :   # 変更なし
            else
                updated=$((updated+1))
            fi
        else
            failed=$((failed+1))
        fi
    done

    # 1つでも取得に失敗したら、安全のため今回は差し替えない（次回に見送り）
    if [ "$failed" -gt 0 ]; then
        echo "   （更新の確認に一部失敗。今のバージョンのまま起動します）"
        rm -rf "$TMP"
        return 0
    fi

    if [ "$updated" -gt 0 ]; then
        echo "   ⬆️ 新しいバージョンが見つかりました。更新しています..."
        for f in $FILES; do
            [ -f "$TMP/$f" ] && cp "$TMP/$f" "$DIR/$f"
        done
        chmod +x "$DIR/setup_common.sh" 2>/dev/null
        echo "   ✅ 最新版になりました。"
    else
        echo "   ✅ すでに最新版です。"
    fi

    date +%s > "$STAMP" 2>/dev/null
    rm -rf "$TMP"
}

echo "========================================================"
echo "  🎧 DJ Video Maker（ブラウザ版）を準備しています..."
echo "========================================================"
echo ""

# ---- フォルダの場所を確定して、まず自動アップデート ----
DIR="$(cd "$(dirname "$0")" && pwd)"
djvm_auto_update "$DIR"
echo ""

# ---- 共通セットアップ部品を読み込む ----
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
