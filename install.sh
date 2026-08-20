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

# 「URLエンコード名<TAB>保存するファイル名」の対。
# ★エンコードを python3 で作ってはいけない。まっさらなMacの python3 は
#   実体が無く、実行すると開発者ツールのインストールを促すだけで結果が空になる。
#   その結果、日本語名のファイルだけURLが空になり404で落ちていた
#   （実例: 2026-08-21 配布先のMacで .command と .txt/.pdf が全滅）。
#   GitHubはNFD正規化で保存しているので、その値を直接埋め込む。
FILES=(
  "dj_maker_core.py	dj_maker_core.py"
  "lipsync_pro.py	lipsync_pro.py"
  "vocal_sync.py	vocal_sync.py"
  "mouth_sync.py	mouth_sync.py"
  "web_server.py	web_server.py"
  "web_ui.html	web_ui.html"
  "setup_common.sh	setup_common.sh"
  "%E3%81%93%E3%82%8C%E3%82%92%E3%82%BF%E3%82%99%E3%83%95%E3%82%99%E3%83%AB%E3%82%AF%E3%83%AA%E3%83%83%E3%82%AF%EF%BC%81.command	これをダブルクリック！.command"
  "DJ_Video_Maker.command	DJ_Video_Maker.command"
  "DJ_Video_Maker_URL.command	DJ_Video_Maker_URL.command"
  "%E4%BF%AE%E5%BE%A9_%E5%88%9D%E5%9B%9E%E3%81%8B%E3%82%89%E3%82%84%E3%82%8A%E7%9B%B4%E3%81%97.command	修復_初回からやり直し.command"
  "%E6%9C%80%E5%88%9D%E3%81%AB%E3%81%93%E3%82%8C%E3%82%92%E5%AE%9F%E8%A1%8C.command	最初にこれを実行.command"
  "%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%88%E3%82%99%E9%9B%86.txt	コマンド集.txt"
  "%E3%81%8B%E3%82%93%E3%81%9F%E3%82%93%E8%AA%AC%E6%98%8E%E6%9B%B8.pdf	かんたん説明書.pdf"
  "%E6%9C%80%E6%96%B0%E7%89%88%E3%81%AB%E3%82%A2%E3%83%83%E3%83%95%E3%82%9A%E3%83%86%E3%82%99%E3%83%BC%E3%83%88.command	最新版にアップデート.command"
)
# 「利用ログ設定手順.txt」は配布しない。あれは開発者（集計する側）向けの
# Apps Script設定手順であり、利用者のフォルダに入れる文書ではないため。

echo "📥 ダウンロード中 → $DEST"
mkdir -p "$DEST"
FAIL=""
for entry in "${FILES[@]}"; do
    enc="${entry%%	*}"      # タブより前＝URL用のエンコード済み名
    name="${entry##*	}"     # タブより後＝保存するファイル名
    # 502等の一時エラーに備えてリトライ。--retry-all-errors で5xxも拾う。
    if ! curl -fsSL --retry 3 --retry-delay 2 --retry-all-errors \
              -o "$DEST/$name" "${BASE}/${enc}"; then
        FAIL="$FAIL $name"
    fi
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
