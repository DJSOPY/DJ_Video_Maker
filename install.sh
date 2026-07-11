#!/bin/bash
# ============================================================
#  DJ Video Maker — ワンライン・インストーラ（ダウンロード担当）
#  仲間に渡す1行：
#    curl -fsSL https://raw.githubusercontent.com/DJSOPY/DJ_Video_Maker/main/install.sh | bash
#
#  ★設計：この install.sh は「取得」だけを行い、
#    パスワード入力やEnter待ちが要る本セットアップは、
#    取得後に新しいTerminalウインドウを開いてそこで実行します。
#    （curl | bash は標準入力がパイプで埋まり、sudoやreadが壊れるため）
# ============================================================
set -u

# ---- ここだけ自分のGitHubに合わせて書き換える ----
GH_USER="DJSOPY"
GH_REPO="DJ_Video_Maker"
GH_BRANCH="main"
# ---------------------------------------------------

BASE="https://raw.githubusercontent.com/${GH_USER}/${GH_REPO}/${GH_BRANCH}"
DEST="$HOME/Desktop/DJ_Video_Maker"

echo "========================================================"
echo "  🎧 DJ Video Maker インストーラ"
echo "     GitHubから一式をダウンロードします"
echo "========================================================"
echo ""

# ---- ネット到達性チェック ----
if ! curl -m 12 -s -o /dev/null -I "https://raw.githubusercontent.com"; then
    echo "❌ GitHubにつながりません。VPNを切る／回線を変えて再実行してください。"
    exit 1
fi

FILES=(
  "dj_maker_core.py" "lipsync_pro.py" "vocal_sync.py" "mouth_sync.py"
  "web_server.py" "web_ui.html" "setup_common.sh"
  "これをダブルクリック！.command" "DJ_Video_Maker.command"
  "DJ_Video_Maker_URL.command" "修復_初回からやり直し.command"
  "最初にこれを実行.command" "コマンド集.txt" "かんたん説明書.pdf"
)

echo "📥 ダウンロード中 → $DEST"
mkdir -p "$DEST"
FAIL=""
for f in "${FILES[@]}"; do
    ok=""
    # macOSの日本語ファイル名は濁点の扱い(NFC/NFD)が混在しうるため、両方＋非エンコードを順に試す
    for variant in \
        "$(python3 -c "import urllib.parse,unicodedata,sys;print(urllib.parse.quote(unicodedata.normalize('NFD',sys.argv[1])))" "$f" 2>/dev/null)" \
        "$(python3 -c "import urllib.parse,unicodedata,sys;print(urllib.parse.quote(unicodedata.normalize('NFC',sys.argv[1])))" "$f" 2>/dev/null)" \
        "$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$f" 2>/dev/null)"
    do
        [ -z "$variant" ] && continue
        if curl -fsSL --retry 2 -o "$DEST/$f" "${BASE}/${variant}"; then ok="1"; break; fi
    done
    [ -z "$ok" ] && FAIL="$FAIL $f"
done

# 実行権限（curl取得なので隔離属性は付かない＝警告が出ない）
chmod +x "$DEST"/*.command "$DEST/setup_common.sh" 2>/dev/null

if [ -n "$FAIL" ]; then
    echo ""
    echo "⚠️ 一部のファイルをダウンロードできませんでした。"
    echo "   ネットが不安定だった可能性があります。"
    echo "   もう一度、さっきの1行を貼り付けて Enter してみてください。"
    echo "   （何度やってもダメなら、この画面を写真に撮って"
    echo "     配布元（@sousouagain）に送ってください）"
    echo ""
    read -p "Enterで閉じる..." _ 2>/dev/null || true
    exit 1
fi

echo "✅ ダウンロード完了！"
echo ""
echo "──────────────────────────────────────────"
echo "  続けて初回セットアップを始めます。"
echo "  新しいターミナル画面が開くので、そこで進めてください。"
echo "  （パスワード入力やEnter待ちがあるため、画面を分けています）"
echo "──────────────────────────────────────────"

# ---- パスワード入力やEnter待ちが要る本番は、新しいTerminalで実行 ----
LAUNCH="$DEST/これをダブルクリック！.command"
if open "$LAUNCH" 2>/dev/null; then
    echo ""
    echo "▶ 新しく開いた画面で準備が進みます（初回は10〜30分）。"
    echo "   ・「Password:」と出たら → Macのパスワードを打って Enter"
    echo "     （画面には出ませんが、ちゃんと打てています）"
    echo "   ・「Press RETURN」と出たら → Enter を1回"
    echo "   ・準備が終わると、ブラウザが自動で開きます。"
    echo ""
    echo "   ※新しく開いた画面は、終わるまで閉じないでください。"
    echo "   （このダウンロード画面の方は、閉じてOKです）"
else
    echo ""
    echo "▶ 準備の画面を自動で開けませんでした。かんたんな手動操作をお願いします："
    echo "   1) デスクトップに「DJ_Video_Maker」フォルダができています。"
    echo "   2) その中の「これをダブルクリック！.command」をダブルクリック。"
    echo "   （それで準備が始まります。初回は10〜30分）"
    open "$DEST" 2>/dev/null || true
fi
echo ""
echo "  ── 次回からの使い方 ──"
echo "  デスクトップの DJ_Video_Maker フォルダの"
echo "  「これをダブルクリック！.command」をダブルクリックするだけです。"
