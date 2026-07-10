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
    enc="$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$f" 2>/dev/null)"
    if ! curl -fsSL --retry 2 -o "$DEST/$f" "${BASE}/${enc}"; then
        curl -fsSL --retry 2 -o "$DEST/$f" "${BASE}/$f" || FAIL="$FAIL $f"
    fi
done

# 実行権限（curl取得なので隔離属性は付かない＝警告が出ない）
chmod +x "$DEST"/*.command "$DEST/setup_common.sh" 2>/dev/null

if [ -n "$FAIL" ]; then
    echo ""
    echo "⚠️ 取得できなかったファイル：$FAIL"
    echo "   GitHubに同名でアップされているか確認して、もう一度1行を実行してください。"
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
# 「これをダブルクリック！.command」を open で起動＝独立したTerminalウインドウで動く
open "$DEST/これをダブルクリック！.command"

echo ""
echo "▶ 新しく開いたターミナル画面で準備が進みます（初回10〜30分）。"
echo "   終わるとブラウザが自動で開きます。この画面は閉じてOKです。"
echo ""
echo "  次回からは、デスクトップの DJ_Video_Maker フォルダにある"
echo "  「これをダブルクリック！.command」をダブルクリックするだけです。"
