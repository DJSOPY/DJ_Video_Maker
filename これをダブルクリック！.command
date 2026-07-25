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

    # ★起動のたびに毎回チェックする。
    #   以前は「24時間に1回」だったため、GitHubを更新してすぐ起動しても
    #   その日はチェック済み扱いでスキップされ、古いコードのまま動いていた。
    #   本体7ファイルは合計1MB未満で、確認と取得は数秒。確実さを優先する。
    #   一時的に自動更新を止めたい時は、同じフォルダに「.no_autoupdate」を置く。
    if [ -f "$DIR/.no_autoupdate" ]; then
        return 0
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

# ---- 利用ログの送信（起動のたびに1回・誰がいつ使ったかを集計先へ）----
# 目的：配った仲間の「最終利用・累計回数・バージョン」を把握する。
#   ・送る内容は最小限：Macのユーザー名・端末名・OS・アプリ版数・日時のみ
#     （曲名や音源など、作った中身は一切送らない）
#   ・集計先URL（下の DJVM_LOG_URL）が空、またはネット無しなら何もしない＝起動に影響なし
#   ・バックグラウンドで投げっぱなし（応答を待たない）ので起動は遅くならない
#   ・止めたい人は、同じフォルダに「.no_usagelog」を置けば送信しない
# ★DJVM_LOG_URL に、Google Apps Script で発行した「ウェブアプリのURL」を貼ってください。
#   （手順は同梱の「利用ログ設定手順.txt」を参照。空のままなら送信機能はオフ。）
DJVM_LOG_URL="https://script.google.com/macros/s/AKfycbx_AEyDq4eAabliKv15_cVsKis7CzgpS3SSR8SO4rm2N_qcsTMi4SOFBWbad23mEWM4/exec"
if [ -n "$DJVM_LOG_URL" ] && [ ! -f "$DIR/.no_usagelog" ]; then
    _u="$(id -un 2>/dev/null)"
    _host="$(scutil --get ComputerName 2>/dev/null || hostname 2>/dev/null)"
    _osv="$(sw_vers -productVersion 2>/dev/null)"
    _ver="$(wc -l < "$DIR/dj_maker_core.py" 2>/dev/null | tr -d ' ')"
    _ts="$(date '+%Y-%m-%d %H:%M:%S')"
    (
      curl -fsS -m 8 -G "$DJVM_LOG_URL" \
        --data-urlencode "user=$_u" \
        --data-urlencode "host=$_host" \
        --data-urlencode "os=$_osv" \
        --data-urlencode "ver=$_ver" \
        --data-urlencode "ts=$_ts" \
        -o /dev/null 2>/dev/null
    ) &
fi

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
