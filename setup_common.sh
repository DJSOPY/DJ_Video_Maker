#!/bin/bash
# ============================================================
#  🔧 DJ Video Maker — 共通セットアップ部品 (setup_common.sh)
#  各 .command から source されて使われます（単体では実行しません）。
#  役割：
#   ① 事前診断（ネット/VPN・Command Line Tools・Homebrew残骸）
#   ② Homebrewインストール（パスワード先行認証つき・対話式）
#   ③ brewが使えない時の保険：ffmpeg / ffprobe / yt-dlp を
#      静的バイナリで直接ダウンロード（パスワード不要・管理者不要）
#   ④ Python環境（venv）とライブラリ
#   ⑤ 最終検証（入っていないのに「準備完了」と言わない）
# ============================================================

# ---- 置き場所 ----
DJVM_HOME="$HOME/.dj_video_maker"
DJVM_BIN="$DJVM_HOME/bin"          # 静的バイナリの置き場（保険ルート）
mkdir -p "$DJVM_HOME" 2>/dev/null

# ---- PATH（Homebrew: Apple Silicon / Intel 両対応 ＋ 静的バイナリ置き場）----
[ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
[ -x /usr/local/bin/brew ]    && eval "$(/usr/local/bin/brew shellenv)"
export PATH="$PATH:$DJVM_BIN"      # brewがあればbrew優先、無ければ静的が効く
export HOMEBREW_NO_ENV_HINTS=1 HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ANALYTICS=1

# ---- 静的バイナリの取得元（2026-07-10 検証済み）----
#  yt-dlp : 公式リリースの固定URL（arm64+Intel両対応のUniversalバイナリ）
#  ffmpeg : martin-riedl.de のスクリプト用固定URL（署名済み・arm64/Intel別）
DJVM_YTDLP_URL="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"
case "$(uname -m)" in
  arm64) DJVM_FFARCH="arm64" ;;
  *)     DJVM_FFARCH="amd64" ;;
esac
DJVM_FFMPEG_URL="https://ffmpeg.martin-riedl.de/redirect/latest/macos/${DJVM_FFARCH}/release/ffmpeg.zip"
DJVM_FFPROBE_URL="https://ffmpeg.martin-riedl.de/redirect/latest/macos/${DJVM_FFARCH}/release/ffprobe.zip"

djvm_pause_exit(){ echo ""; read -p "Enterで閉じる..."; exit 1; }

# ============================================================
# ⓪ 軽い残骸掃除（毎回の起動時に実行して安全なものだけ）
#    ※重い掃除（venv作り直し・brew reinstall・静的バイナリ入れ直し）は
#      修復_初回からやり直し.command だけが行う
# ============================================================
djvm_light_cleanup(){
    # openssl@1.1 の壊れたリンク（"is not a valid keg" エラーの原因）
    rm -f /usr/local/opt/openssl@1.1 /opt/homebrew/opt/openssl@1.1 2>/dev/null
    # Pythonキャッシュの死骸（.pyを差し替えても古い挙動が出るのを防ぐ）
    find "$(pwd)" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    find "$(pwd)" -name "*.pyc" -delete 2>/dev/null
    # brewのゴミ掃除（brewがある時だけ・静かに）
    command -v brew &>/dev/null && brew cleanup 2>/dev/null
    return 0
}

# ============================================================
# ① 事前診断
# ============================================================

# --- ①-1 ネット接続（Wi-Fi / VPN / 施設のブロック検知）---
djvm_check_network(){
    echo "🌐 [診断] ネット接続を確認しています..."
    local ok=""
    for url in "https://github.com" "https://raw.githubusercontent.com"; do
        if curl -m 12 -s -o /dev/null -I "$url" 2>/dev/null; then ok="1"; break; fi
    done
    # 1回目失敗ならすこし待って再試行（瞬断対策）
    if [ -z "$ok" ]; then
        sleep 3
        curl -m 12 -s -o /dev/null -I "https://github.com" 2>/dev/null && ok="1"
    fi
    if [ -z "$ok" ]; then
        echo ""
        echo "❌ ダウンロードに必要なサイト（GitHub）につながりません。"
        echo "   よくある原因と対処："
        echo "   ・Wi-Fiが切れている／不安定 → Wi-Fiを一度切って入れ直す"
        echo "   ・VPNアプリを使っている   → VPNをオフにして、もう一度実行"
        echo "   ・お店/会社/学校のWi-Fi   → ダウンロードがブロックされている"
        echo "     ことがあります → iPhoneのテザリング等、別の回線で試す"
        echo ""
        echo "   回線を直したら、もう一度このアイコンをダブルクリックしてください。"
        djvm_pause_exit
    fi
    echo "   ✓ ネット接続OK"
}

# --- ①-2 Command Line Tools（無い／壊れている検知）---
djvm_check_clt(){
    # brewが既に動いているなら CLT も生きている（この確認はスキップ）
    command -v brew &>/dev/null && return 0
    local p
    p="$(xcode-select -p 2>/dev/null)"
    if [ -n "$p" ] && [ -d "$p" ]; then
        return 0    # 正常
    fi
    if [ -n "$p" ] && [ ! -d "$p" ]; then
        echo ""
        echo "⚠️ [診断] macOSの開発ツール(Command Line Tools)が壊れています。"
        echo "   （登録されている場所 $p が実在しません）"
        echo "   ターミナルで次の2行を順に実行して直してください："
        echo "       sudo rm -rf $p"
        echo "       xcode-select --install"
        echo "   →「インストール」ボタンのダイアログが出るので進めて、"
        echo "     終わったらもう一度このアイコンをダブルクリックしてください。"
        djvm_pause_exit
    fi
    # 未インストール → インストールダイアログをこちらから出してあげる
    echo ""
    echo "📦 [診断] macOSの開発ツール(Command Line Tools)が未インストールです。"
    echo "   今からインストール画面（ダイアログ）を出します。"
    echo "   →「インストール」を押して完了までお待ちください（10〜20分）。"
    echo "   → 完了したら、もう一度このアイコンをダブルクリックしてください。"
    xcode-select --install 2>/dev/null
    djvm_pause_exit
}

# --- ①-3 Homebrewの残骸（コマンドは無いのにフォルダだけ残っている）---
djvm_check_brew_leftover(){
    command -v brew &>/dev/null && return 0
    if [ -d /opt/homebrew/Cellar ] || [ -d /usr/local/Homebrew ]; then
        echo ""
        echo "⚠️ [診断] 以前のHomebrewの残骸が見つかりました（コマンドは使えない状態）。"
        echo "   このまま上書きインストールを試みます。それで直ることが多いですが、"
        echo "   失敗する場合は同梱の『修復_初回からやり直し.command』を実行してください。"
        echo ""
    fi
}

# ============================================================
# ② Homebrew インストール（管理者のみ・パスワード先行認証つき）
# ============================================================
djvm_install_homebrew(){
    echo ""
    echo "┌──────────────────────────────────────────────────┐"
    echo "│ 📦 初回準備：基本ツール(Homebrew)を入れます        │"
    echo "│                                                  │"
    echo "│ ⚠️ この先、あなたの操作が必要な場面が 2回 あります │"
    echo "│                                                  │"
    echo "│  1) Password: と出たら                            │"
    echo "│     → Macのログインパスワードを打って Enter       │"
    echo "│       （画面には何も表示されませんが打てています）│"
    echo "│                                                  │"
    echo "│  2) Press RETURN/ENTER と出たら                   │"
    echo "│     → そのまま Enter キーを1回押す                │"
    echo "│                                                  │"
    echo "│ ⚠️ 完了まで、このウィンドウは絶対に閉じないで！    │"
    echo "│    （閉じてしまっても、修復.commandでやり直せます）│"
    echo "└──────────────────────────────────────────────────┘"
    echo ""
    # --- パスワードを「先に」済ませる（間違いならこの場で分かる）---
    echo "🔑 まずパスワードの確認です。Password: に続けて入力して Enter"
    echo "   （Macにログインする時のパスワードです。画面には表示されません）"
    if ! sudo -v; then
        echo ""
        echo "❌ パスワードの確認に失敗しました。"
        echo "   ・打つのは「Macのログインパスワード」です（Apple IDではありません）"
        echo "   ・画面に何も出なくても入力できています。ゆっくり打って Enter"
        echo "   もう一度このアイコンをダブルクリックしてやり直してください。"
        djvm_pause_exit
    fi
    echo "   ✓ パスワードOK。インストールを開始します（10分前後）..."
    # --- 認証を切らさないための番兵（長いインストール中の再入力を防ぐ）---
    ( while true; do sudo -n true 2>/dev/null; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) &
    local KEEPALIVE=$!
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    kill "$KEEPALIVE" 2>/dev/null
    [ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
    [ -x /usr/local/bin/brew ]    && eval "$(/usr/local/bin/brew shellenv)"
    if command -v brew &>/dev/null; then
        echo "   ✓ Homebrew インストール完了"
        return 0
    fi
    echo "⚠️ Homebrewが入りませんでした → パスワード不要の直接ダウンロード方式に切り替えます"
    return 1
}

# ============================================================
# ③ 保険ルート：静的バイナリを直接ダウンロード（brew不要・sudo不要）
# ============================================================
djvm_dl(){  # djvm_dl <URL> <保存先>  （リトライ1回つき）
    curl -L --fail -m 600 --retry 2 -o "$2" "$1" 2>/dev/null || \
    curl -L --fail -m 600 -o "$2" "$1"
}

djvm_install_static_ytdlp(){
    echo "📥 yt-dlp を直接ダウンロード中（約40MB）..."
    mkdir -p "$DJVM_BIN"
    if ! djvm_dl "$DJVM_YTDLP_URL" "$DJVM_BIN/yt-dlp.tmp"; then
        echo "   ❌ yt-dlp のダウンロードに失敗しました（ネット接続を確認）"; return 1
    fi
    mv -f "$DJVM_BIN/yt-dlp.tmp" "$DJVM_BIN/yt-dlp"
    chmod +x "$DJVM_BIN/yt-dlp"
    xattr -c "$DJVM_BIN/yt-dlp" 2>/dev/null
    if ! "$DJVM_BIN/yt-dlp" --version >/dev/null 2>&1; then
        codesign -s - --force "$DJVM_BIN/yt-dlp" 2>/dev/null
        "$DJVM_BIN/yt-dlp" --version >/dev/null 2>&1 || { echo "   ❌ yt-dlp が起動できません"; return 1; }
    fi
    echo "   ✓ yt-dlp $("$DJVM_BIN/yt-dlp" --version 2>/dev/null)"
}

djvm_install_static_ff(){   # djvm_install_static_ff <ffmpeg|ffprobe> <URL>
    local name="$1" url="$2"
    echo "📥 $name を直接ダウンロード中..."
    mkdir -p "$DJVM_BIN"
    local tmp="$DJVM_HOME/_dl_$name"
    rm -rf "$tmp"; mkdir -p "$tmp"
    if ! djvm_dl "$url" "$tmp/$name.zip"; then
        echo "   ❌ $name のダウンロードに失敗しました"; rm -rf "$tmp"; return 1
    fi
    if ! unzip -o -q "$tmp/$name.zip" -d "$tmp"; then
        echo "   ❌ $name の展開に失敗しました"; rm -rf "$tmp"; return 1
    fi
    # zip内のどこにあっても本体を探して所定の場所へ
    local bin
    bin="$(find "$tmp" -type f -name "$name" | head -1)"
    if [ -z "$bin" ]; then echo "   ❌ $name 本体がzip内に見つかりません"; rm -rf "$tmp"; return 1; fi
    mv -f "$bin" "$DJVM_BIN/$name"
    chmod +x "$DJVM_BIN/$name"
    xattr -c "$DJVM_BIN/$name" 2>/dev/null
    rm -rf "$tmp"
    if ! "$DJVM_BIN/$name" -version >/dev/null 2>&1; then
        codesign -s - --force "$DJVM_BIN/$name" 2>/dev/null
        "$DJVM_BIN/$name" -version >/dev/null 2>&1 || { echo "   ❌ $name が起動できません"; return 1; }
    fi
    echo "   ✓ $name OK"
}

djvm_install_static_all(){
    echo ""
    echo "🧰 パスワード不要の直接ダウンロード方式でツールを入れます..."
    local ng=""
    command -v ffmpeg  &>/dev/null || djvm_install_static_ff ffmpeg  "$DJVM_FFMPEG_URL"  || ng="1"
    command -v ffprobe &>/dev/null || djvm_install_static_ff ffprobe "$DJVM_FFPROBE_URL" || ng="1"
    command -v yt-dlp  &>/dev/null || djvm_install_static_ytdlp || ng="1"
    [ -z "$ng" ]
}

# ============================================================
# ツール確保のまとめ（brew優先 → だめなら静的フォールバック）
# ============================================================
djvm_ensure_tools(){
    # --- Homebrewが無い場合 ---
    if ! command -v brew &>/dev/null; then
        djvm_check_brew_leftover
        if groups | grep -qw admin; then
            djvm_install_homebrew   # 失敗しても続行（下で静的にフォールバック）
        else
            echo ""
            echo "ℹ️ このアカウントは管理者ではないため、Homebrewは使わず"
            echo "   パスワード不要の直接ダウンロード方式で準備します。"
        fi
    fi
    # --- brewが使えるなら従来通り ---
    if command -v brew &>/dev/null; then
        if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
            echo "📦 ffmpeg をインストール中..."
            brew install ffmpeg || {
                rm -f /usr/local/opt/openssl@1.1 /opt/homebrew/opt/openssl@1.1 2>/dev/null
                brew uninstall --ignore-dependencies openssl@1.1 2>/dev/null
                brew cleanup 2>/dev/null; brew update 2>/dev/null
                brew install ffmpeg
            }
        fi
        if ! command -v yt-dlp &>/dev/null; then
            echo "📦 yt-dlp をインストール中..."
            brew install yt-dlp || {
                rm -f /usr/local/opt/openssl@1.1 /opt/homebrew/opt/openssl@1.1 2>/dev/null
                brew uninstall --ignore-dependencies openssl@1.1 2>/dev/null
                brew cleanup 2>/dev/null
                brew install yt-dlp
            }
        else
            # YouTubeの仕様変更対策：週1回だけ自動アップデート
            local STAMP="$DJVM_HOME/ytdlp_updated"
            if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(stat -f %m "$STAMP" 2>/dev/null || echo 0) )) -gt 604800 ]; then
                echo "🔄 yt-dlp を最新版に更新中（週1回の自動チェック）..."
                if [ -x "$DJVM_BIN/yt-dlp" ] && [ "$(command -v yt-dlp)" = "$DJVM_BIN/yt-dlp" ]; then
                    djvm_install_static_ytdlp
                else
                    brew upgrade yt-dlp 2>/dev/null
                fi
                touch "$STAMP"
            fi
        fi
    fi
    # --- まだ足りないものがあれば静的フォールバック ---
    if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null || ! command -v yt-dlp &>/dev/null; then
        djvm_install_static_all
    fi
    # --- 静的yt-dlpの週1更新（brew無し運用のケース）---
    if ! command -v brew &>/dev/null && [ -x "$DJVM_BIN/yt-dlp" ]; then
        local STAMP="$DJVM_HOME/ytdlp_updated"
        if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(stat -f %m "$STAMP" 2>/dev/null || echo 0) )) -gt 604800 ]; then
            echo "🔄 yt-dlp を最新版に更新中（週1回の自動チェック）..."
            djvm_install_static_ytdlp && touch "$STAMP"
        fi
    fi
    # --- 最終確認 ---
    local miss=""
    for t in ffmpeg ffprobe yt-dlp; do command -v "$t" &>/dev/null || miss="$miss $t"; done
    if [ -n "$miss" ]; then
        echo ""
        echo "❌ 次のツールが準備できませんでした：$miss"
        echo "   ・回線を変えて（VPNオフ／テザリング等）もう一度実行"
        echo "   ・それでもダメなら『修復_初回からやり直し.command』を実行"
        echo "   💬 それでも直らなければ、この画面を写真に撮って @sousouagain へ"
        djvm_pause_exit
    fi
}

# ============================================================
# ④ Python環境（venv）とライブラリ
# ============================================================
djvm_setup_python(){    # 引数: full=重いAIライブラリも入れる / lite=静かに試すだけ
    local mode="${1:-full}"
    VENV="$HOME/.dj_video_maker_env"
    PYTHON_CMD="$VENV/bin/python3"

    # ============================================================
    #  対応Python(3.12優先)で必ずvenvを用意する（全パターン自動対応）
    #  - 3.14等で作られた壊れたvenvは問答無用で削除
    #  - 3.12/3.13/3.11/3.10 を探す → 無ければ brew で 3.12 を入れる
    #  - brew自体が無ければ brew を入れてから 3.12 を入れる
    #  - 3.14系のシステムpythonには絶対フォールバックしない
    # ============================================================
    _djvm_is_supported_ver(){   # 引数のpython実行体が 3.10〜3.13 か
        local pybin="$1" v
        v="$("$pybin" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
        case "$v" in 3.10|3.11|3.12|3.13) return 0;; *) return 1;; esac
    }

    # --- 既存venvが対応外(3.14等)なら削除して作り直す ---
    if [ -x "$PYTHON_CMD" ]; then
        if ! _djvm_is_supported_ver "$PYTHON_CMD"; then
            local badver
            badver="$("$PYTHON_CMD" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
            echo "🔁 現在のPython($badver)は音声解析ライブラリ非対応のため、対応版で作り直します..."
            rm -rf "$VENV"
        fi
    fi

    if [ ! -x "$PYTHON_CMD" ]; then
        echo "🐍 Python環境を構築中（対応版のPythonを用意します）..."

        # 対応版Pythonを探す（3.12最優先→3.13→3.11→3.10）。3.14系は候補にしない。
        _djvm_find_py(){
            local cand dir
            for cand in python3.12 python3.13 python3.11 python3.10; do
                for dir in /opt/homebrew/bin /usr/local/bin /usr/bin; do
                    [ -x "$dir/$cand" ] && { echo "$dir/$cand"; return 0; }
                done
                command -v "$cand" &>/dev/null && { command -v "$cand"; return 0; }
            done
            return 1
        }

        BASE_PY="$(_djvm_find_py)"

        # 見つからなければ brew で 3.12 を入れる（brewが無ければ brew から入れる）
        if [ -z "$BASE_PY" ]; then
            if ! command -v brew &>/dev/null; then
                echo "📦 まず Homebrew を用意します（初回のみ・パスワードを求められます）..."
                if groups | grep -qw admin; then
                    # sudo先行認証（見えないが入力できている旨は事前に案内済み）
                    sudo -v 2>/dev/null
                    ( while true; do sudo -n true 2>/dev/null; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) &
                    local _ka=$!
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/null
                    kill "$_ka" 2>/dev/null
                    [ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
                    [ -x /usr/local/bin/brew ]    && eval "$(/usr/local/bin/brew shellenv)"
                fi
            fi
            if command -v brew &>/dev/null; then
                echo "📦 対応版の Python(3.12) を用意しています（数分かかります）..."
                brew install python@3.12 2>/dev/null
                for dir in /opt/homebrew/bin /usr/local/bin; do
                    [ -x "$dir/python3.12" ] && { BASE_PY="$dir/python3.12"; break; }
                done
            fi
        fi

        # まだ無ければ、システムのpython3が“対応版の時だけ”最後の手段に
        if [ -z "$BASE_PY" ] && command -v python3 &>/dev/null && _djvm_is_supported_ver "$(command -v python3)"; then
            BASE_PY="$(command -v python3)"
        fi

        if [ -z "$BASE_PY" ]; then
            echo ""
            echo "❌ 音声解析に使える Python(3.10〜3.13) を用意できませんでした。"
            echo "   お手数ですが、ターミナルで次を実行してから、もう一度お試しください："
            echo "       brew install python@3.12"
            echo "   （Homebrew未導入なら、先に『修復_初回からやり直し.command』を実行）"
            djvm_pause_exit
        fi

        echo "   使用するPython: $("$BASE_PY" --version 2>&1)"
        "$BASE_PY" -m venv "$VENV"

        # venvが3.14等になっていないか最終ガード（万一の取り違え防止）
        if [ -x "$PYTHON_CMD" ] && ! _djvm_is_supported_ver "$PYTHON_CMD"; then
            echo "⚠️ 作成したPython環境が対応版になりませんでした。作り直しの上、中断します。"
            rm -rf "$VENV"; djvm_pause_exit
        fi
        if [ ! -x "$PYTHON_CMD" ]; then
            echo "❌ Python環境の作成に失敗しました。"
            echo "   ターミナルで xcode-select --install を実行して開発ツールを入れ、"
            echo "   完了後にもう一度このアイコンをダブルクリックしてください。"
            djvm_pause_exit
        fi
    fi
    if ! "$PYTHON_CMD" -c "import numpy, scipy, mutagen, PIL, librosa, fastdtw" &>/dev/null; then
        echo "📦 基本ライブラリをインストール中（初回・数分）..."
        # pipを最新化（古いpipは新しい完成品(wheel)を認識できずビルドに落ちるため必須）
        "$PYTHON_CMD" -m pip install --quiet --no-input --upgrade pip setuptools wheel
        # ★Intel Mac対策：librosaが依存する numba/llvmlite を、
        #   Intel(x86_64)でもApple Siliconでも完成品(wheel)が確実にあるバージョンに固定して先に入れる。
        #   （最新のllvmliteはIntel Mac用の完成品macOS wheelが無く、ビルドに落ちるため）
        #   numba 0.61.2 → llvmlite 0.44.0：両アーキで cp310〜cp313 の完成品あり。
        "$PYTHON_CMD" -m pip install --only-binary=:all: --no-input \
              "llvmlite==0.44.0" "numba==0.61.2" 2>/tmp/djvm_pip.log \
          || "$PYTHON_CMD" -m pip install --only-binary=:all: --no-input "numba<0.62" 2>>/tmp/djvm_pip.log
        # 【グループA】重いC拡張ライブラリ＝完成品(wheel)のみ許可（ビルド地獄を防ぐ）。
        #   これらは両アーキ×cp310〜cp313で完成品があることを確認済み。
        "$PYTHON_CMD" -m pip install --only-binary=:all: --no-input \
              numpy scipy librosa 2>>/tmp/djvm_pip.log
        # 【グループB】軽い/純Python寄り＝完成品が無くてもソースでOK（コンパイラ不要か軽微）。
        #   fastdtwは全アーキで新しめの完成品が無いが、ソースで問題なく入る（確認済み）。
        "$PYTHON_CMD" -m pip install --no-input \
              mutagen Pillow fastdtw 2>>/tmp/djvm_pip.log
        # --- 本当に使えるか最終確認（必須のものだけ厳しくチェック）---
        if ! "$PYTHON_CMD" -c "import numpy, scipy, mutagen, librosa" &>/dev/null; then
            echo ""
            echo "❌ 音楽解析に必要なライブラリが用意できませんでした。"
            # 原因を分かりやすく振り分けて案内
            pyver="$("$PYTHON_CMD" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
            if grep -qi "llvmlite\|numba\|Failed building wheel" /tmp/djvm_pip.log 2>/dev/null; then
                echo "   原因：この環境向けの音声解析部品(llvmlite/numba)の準備に失敗しました。"
                echo "   ネット接続（VPNオフ・別回線）を確認して、もう一度お試しください。"
                echo "   💬 何度も失敗する場合は、この画面を写真に撮って @sousouagain へ送ってください。"
            else
                echo "   ネット接続を確認して、もう一度お試しください（VPNオフ／別回線も有効）。"
                echo "   💬 何度も失敗する場合は、この画面を写真に撮って @sousouagain へ送ってください。"
            fi
            djvm_pause_exit
        fi
    fi
    if [ "$mode" = "full" ]; then
        if ! "$PYTHON_CMD" -c "import torch, demucs" &>/dev/null; then
            echo ""
            echo "🧠 高精度リップシンク用のAIを準備しています。"
            echo "   数GBのダウンロードがあり、回線によっては 10〜30分 かかります。"
            echo "   ☕️ 進捗バーが動いている間はそのままお待ちください（初回だけ・閉じないで）"
            echo ""
            "$PYTHON_CMD" -m pip install --no-input demucs \
              || "$PYTHON_CMD" -m pip install --no-input demucs \
              || echo "   （高精度ライブラリは入りませんでした → 従来方式で動きます）"
        fi
        if ! "$PYTHON_CMD" -c "import transformers, torchaudio" &>/dev/null; then
            "$PYTHON_CMD" -m pip install --no-input transformers torchaudio \
              || "$PYTHON_CMD" -m pip install --no-input transformers torchaudio \
              || echo "   （HuBERT用ライブラリは入りませんでした → MFCC/従来方式で動きます）"
        fi
        if ! "$PYTHON_CMD" -c "import faster_whisper" &>/dev/null; then
            "$PYTHON_CMD" -m pip install --no-input faster-whisper \
              || "$PYTHON_CMD" -m pip install --no-input faster-whisper \
              || echo "   （Whisper単語アライメントは入りませんでした → その段はスキップして動きます）"
        fi
        if [ "${DJVM_INSTALL_WHISPERX:-0}" = "1" ]; then
            if ! "$PYTHON_CMD" -c "import whisperx" &>/dev/null; then
                "$PYTHON_CMD" -m pip install --no-input whisperx \
                  || echo "   （WhisperXは入りませんでした → faster-whisperで単語アライメントします）"
            fi
        fi
        # ★mediapipeは 0.10.31以降のmacOS wheelから旧solutions API(face_mesh)が
        #   削除された。DJ Video Makerの口元解析は face_mesh を使い、新Tasks APIは
        #   macOSでMetal初期化に失敗してプロセスごと落ちる環境があるため使わない。
        #   → 口元解析が動く最後のバージョン 0.10.21 に固定する(universal2＝Intel/Silicon両対応)。
        #   solutionsが無い版が既に入っている場合は入れ直す。
        if ! "$PYTHON_CMD" -c "import mediapipe.python.solutions.face_mesh, cv2" &>/dev/null; then
            echo "   🔧 口元解析ライブラリ(mediapipe)を対応版に調整中..."
            # mediapipeは jax / jaxlib を必須依存に書いているが、口元解析
            # (legacy solutions face_mesh)はjaxを一切importしない（実測確認済み）。
            # jaxlibにはIntel Mac(x86_64)版wheelが無く、依存解決が延々と過去へ遡って
            # 失敗/長時間化するため、--no-deps で入れて必要な物だけ自分で足す。
            # これでIntel/Apple Siliconのどちらでも同じ手順で確実に入る。
            "$PYTHON_CMD" -m pip install --no-input --force-reinstall --no-deps "mediapipe==0.10.21" 2>>/tmp/djvm_pip.log
            "$PYTHON_CMD" -m pip install --no-input "numpy<2" "protobuf<5,>=4.25.3" absl-py attrs flatbuffers sounddevice sentencepiece matplotlib opencv-python opencv-contrib-python 2>>/tmp/djvm_pip.log || echo "   （口の動き解析ライブラリは入りませんでした → 人物カットは安全のため背景に置き換わります）"
        fi
        if ! "$PYTHON_CMD" -c "import mediapipe.python.solutions.face_mesh" &>/dev/null; then
            echo "   ⚠️ 口元解析(mediapipe face_mesh)が使えません。"
            echo "      → 人物が映るカットは安全のため背景に置き換わります。"
            echo "      復旧するには次を実行してください:"
            echo "      $PYTHON_CMD -m pip install --force-reinstall \"mediapipe==0.10.21\""
        fi
    else
        "$PYTHON_CMD" -c "import mediapipe.python.solutions.face_mesh, cv2" 2>/dev/null || { "$PYTHON_CMD" -m pip install --no-input --force-reinstall --no-deps "mediapipe==0.10.21" 2>/dev/null; "$PYTHON_CMD" -m pip install --no-input "numpy<2" "protobuf<5,>=4.25.3" absl-py attrs flatbuffers sounddevice sentencepiece matplotlib opencv-python opencv-contrib-python 2>/dev/null; } || true
        "$PYTHON_CMD" -c "import demucs" 2>/dev/null || "$PYTHON_CMD" -m pip install --no-input demucs 2>/dev/null || true
        "$PYTHON_CMD" -c "import faster_whisper" 2>/dev/null || "$PYTHON_CMD" -m pip install --no-input faster-whisper 2>/dev/null || true
    fi
    export DJVM_PYTHON="$PYTHON_CMD"
}

# ============================================================
# ⑤ 最終検証
# ============================================================
djvm_verify(){
    local miss=""
    for t in ffmpeg ffprobe yt-dlp; do command -v "$t" &>/dev/null || miss="$miss $t"; done
    "$PYTHON_CMD" -c "import numpy, scipy, librosa" 2>/dev/null || miss="$miss pythonライブラリ"
    if [ -n "$miss" ]; then
        echo ""
        echo "❌ セットアップが完了していません。次が使えません：$miss"
        echo "   ネット接続（VPNオフ・別回線も試す）を確認して、"
        echo "   もう一度このアイコンをダブルクリックしてください。"
        echo "   何度も失敗する場合は『修復_初回からやり直し.command』を実行してください。"
        echo ""
        echo "   💬 それでも直らない時は、この画面を写真に撮って"
        echo "      配布元（@sousouagain）に送ってください。すぐ直します。"
        djvm_pause_exit
    fi
}

# ============================================================
# ひとまとめ（各.commandはこれを呼ぶだけ）
#   djvm_full_setup full   … ターミナル版（AIライブラリも入れる）
#   djvm_full_setup lite   … サーバー版（AIは静かに試すだけ）
# ============================================================
djvm_full_setup(){
    djvm_light_cleanup
    djvm_check_network
    djvm_check_clt
    djvm_ensure_tools
    djvm_setup_python "${1:-full}"
    djvm_verify
}
