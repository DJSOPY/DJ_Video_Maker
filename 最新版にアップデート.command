#!/bin/bash
# ============================================================
#  🔄 DJ Video Maker — 最新版にアップデート
#  これをダブルクリックするだけで、最新版に入れ替わります。
#  （ターミナルを触る必要はありません）
#
#  使い方：
#   ・古いバージョンを使っている人に、このファイルを1個だけ送ってください。
#   ・受け取った人は、ダウンロードしてダブルクリックするだけ。
#   ・一度これで更新すれば、以降は本体を開くたびに自動で最新になります
#     （次回からはこのファイルを使う必要はありません）。
# ============================================================
cd "$(dirname "$0")"
# ネット経由で受け取ったファイルの隔離属性を外す（「開発元が未確認」警告の回避）
xattr -dr com.apple.quarantine "$(pwd)" 2>/dev/null
clear
echo "========================================================"
echo "  🔄 DJ Video Maker を最新版にアップデートします"
echo "========================================================"
echo ""

# ネット接続の確認（GitHubに繋がるか）
if ! curl -m 12 -s -o /dev/null -I "https://raw.githubusercontent.com" 2>/dev/null; then
    echo "❌ インターネットに繋がっていないようです。"
    echo "   Wi-Fiなどネット接続を確認して、もう一度ダブルクリックしてください。"
    echo ""
    read -p "Enterで閉じる..."; exit 1
fi

echo "📥 最新版をGitHubから取得して入れ替えます..."
echo "   （デスクトップに「DJ_Video_Maker」フォルダを最新の状態で作り直します）"
echo ""

# 公式の配布スクリプトをそのまま実行（＝新規インストールと同じ最新一式が入る）。
# curl | bash は read/sudo が壊れるため、ファイルに落としてから bash で実行する。
TMP_INSTALLER="$(mktemp -t djvm_install).sh"
if curl -fsSL --retry 2 -o "$TMP_INSTALLER" \
    "https://raw.githubusercontent.com/DJSOPY/DJ_Video_Maker/main/install.sh"; then
    bash "$TMP_INSTALLER"
    _rc=$?
    rm -f "$TMP_INSTALLER" 2>/dev/null
    if [ "$_rc" = "0" ]; then
        echo ""
        echo "========================================================"
        echo "  ✅ アップデート完了！"
        echo "     デスクトップの「DJ_Video_Maker」フォルダを開いて、"
        echo "     中の「これをダブルクリック！.command」から起動してください。"
        echo ""
        echo "  ※以降は本体を開くたびに自動で最新版になります。"
        echo "     このアップデート用ファイルは、もう使わなくてOKです。"
        echo "========================================================"
        # 完了したらフォルダを開いてあげる
        [ -d "$HOME/Desktop/DJ_Video_Maker" ] && open "$HOME/Desktop/DJ_Video_Maker" 2>/dev/null
    else
        echo ""
        echo "⚠️ 途中で問題が起きたようです。もう一度ダブルクリックして試すか、"
        echo "   うまくいかない場合は配布元（DJ SOPY）に連絡してください。"
    fi
else
    rm -f "$TMP_INSTALLER" 2>/dev/null
    echo "❌ 最新版の取得に失敗しました。"
    echo "   少し時間をおいて、もう一度ダブルクリックしてください。"
fi
echo ""
read -p "Enterで閉じる..."
