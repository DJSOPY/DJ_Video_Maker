#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#  DJ Video Maker — ローカルWebサーバー（ブラウザで操作する版）
#  既存エンジン(dj_maker_core.py)を無改造で裏で動かし、
#  ブラウザから 曲アップ→オプション→作成→ダウンロード まで行う。
#  追加インストール不要（Python標準ライブラリのみ）。
# ============================================================
import http.server, socketserver, subprocess, threading, json, os, sys, re, time, uuid
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

HERE = Path(__file__).resolve().parent
CORE = HERE / "dj_maker_core.py"
CONFIG_DIR = Path.home() / ".dj_video_maker"
CONFIG_FILE = CONFIG_DIR / "config.json"
WORK = CONFIG_DIR / "web_jobs"
WORK.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("DJVM_PORT", "8765"))

# Cookie設定を事前に用意（無いと起動時に「どのブラウザ？」で止まるため）
def normalize_youtube_url(url):
    """YouTube URLから動画IDだけを抽出しクリーンな単一動画URLにする。再生リスト等を除去。"""
    import re
    url = (url or "").strip()
    vid = None
    m = re.search(r'youtu\.be/([A-Za-z0-9_-]{11})', url)
    if m: vid = m.group(1)
    if not vid:
        m = re.search(r'[?&]v=([A-Za-z0-9_-]{11})', url)
        if m: vid = m.group(1)
    if not vid:
        m = re.search(r'/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})', url)
        if m: vid = m.group(1)
    return f"https://www.youtube.com/watch?v={vid}" if vid else None


def ensure_config(browser="chrome"):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG_FILE.exists():
        try: cfg = json.loads(CONFIG_FILE.read_text())
        except Exception: cfg = {}
    if "cookies_browser" not in cfg:
        cfg["cookies_browser"] = browser
        CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    return cfg

def set_browser(browser):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG_FILE.exists():
        try: cfg = json.loads(CONFIG_FILE.read_text())
        except Exception: cfg = {}
    cfg["cookies_browser"] = browser
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

# ---- multipart/form-data を最小パース（ファイル＋テキスト項目）----
def parse_multipart(body, boundary):
    parts = body.split(b"--" + boundary)
    files, fields = [], {}
    for p in parts:
        if not p or p in (b"--\r\n", b"--"): continue
        if b"\r\n\r\n" not in p: continue
        head, data = p.split(b"\r\n\r\n", 1)
        if data.endswith(b"\r\n"): data = data[:-2]
        head_s = head.decode("utf-8", "replace")
        m = re.search(r'name="([^"]*)"', head_s)
        if not m: continue
        name = m.group(1)
        fn = re.search(r'filename="([^"]*)"', head_s)
        if fn and fn.group(1):
            files.append((name, fn.group(1), data))
        else:
            fields[name] = data.decode("utf-8", "replace").strip()
    return files, fields


JOBS = {}   # job_id -> {"log":[...], "done":bool, "outputs":[...], "proc":..}

REPAIR_SH = r'''
set +e
echo "🛠 修復を開始します（パスワード不要の範囲）..."
[ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
[ -x /usr/local/bin/brew ]    && eval "$(/usr/local/bin/brew shellenv)"
export PATH="$PATH:$HOME/.dj_video_maker/bin"
export HOMEBREW_NO_ENV_HINTS=1 HOMEBREW_NO_AUTO_UPDATE=1
echo "🧹 [1/4] 壊れた古い残骸を掃除..."
rm -f /usr/local/opt/openssl@1.1 /opt/homebrew/opt/openssl@1.1 2>/dev/null
brew uninstall --ignore-dependencies openssl@1.1 2>/dev/null
brew cleanup 2>/dev/null
echo "🎬 [2/4] ffmpeg を入れ直し..."
if command -v brew >/dev/null 2>&1; then
  brew reinstall ffmpeg 2>/dev/null || brew install ffmpeg
fi
echo "📥 [3/4] yt-dlp を入れ直し＋最新化..."
if command -v brew >/dev/null 2>&1; then
  brew reinstall yt-dlp 2>/dev/null || brew install yt-dlp
  brew upgrade yt-dlp 2>/dev/null
fi
# brewで揃わない分は、パスワード不要の直接ダウンロードで補う
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1 || ! command -v yt-dlp >/dev/null 2>&1; then
  if [ -f "$DJVM_SETUP" ]; then
    . "$DJVM_SETUP"
    djvm_install_static_all
  else
    echo "⚠️ setup_common.sh が見つからないため直接ダウンロードを実行できません"
  fi
fi
echo "🐍 [4/4] Pythonライブラリを確認..."
"$DJVM_PY" -m pip install --no-input --upgrade numpy scipy mutagen Pillow librosa fastdtw 2>/dev/null
echo "🔍 確認中..."
NG=""
command -v ffmpeg  >/dev/null 2>&1 || NG="$NG ffmpeg"
command -v ffprobe >/dev/null 2>&1 || NG="$NG ffprobe"
command -v yt-dlp  >/dev/null 2>&1 || NG="$NG yt-dlp"
if [ -z "$NG" ]; then
  echo "✅ 修復完了！ ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3) / yt-dlp $(yt-dlp --version 2>/dev/null)"
  echo "   このまま曲を作れます。"
else
  echo "⚠️ まだ次が入っていません:$NG"
  echo "   同梱の『修復_初回からやり直し.command』を実行してください（回線を変えるのも有効です）。"
fi
'''

def run_repair(job_id, py):
    job = JOBS[job_id]
    env = dict(os.environ); env["PYTHONUNBUFFERED"]="1"; env["DJVM_PY"]=py
    env["DJVM_SETUP"] = str(HERE / "setup_common.sh")
    try:
        proc = subprocess.Popen(["/bin/bash","-c",REPAIR_SH], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        job["proc"] = proc
        for line in proc.stdout:
            job["log"].append(line.rstrip("\n"))
        proc.wait()
    except Exception as e:
        job["log"].append(f"❌ 修復エラー: {e}")
    finally:
        job["done"] = True

def run_engine(job, music_paths, out_dir, extend, py, urls=None, audio_cap="320"):
    """urls=None → 自動選択モード。urls=[...] → URLモード（曲順にURLを流す）。"""
    a = [str(p) for p in music_paths] + [""] + [str(out_dir)]
    a += (["y","",""] if extend else [""])
    if urls:                       # URLモード：曲順にURLを1行ずつ
        a += [u for u in urls]
    answers = "\n".join(a) + "\n"
    env = dict(os.environ); env["PYTHONUNBUFFERED"]="1"; env["PYTHONIOENCODING"]="utf-8"
    env["DJVM_WEB"] = "1"
    env["DJVM_AUDIO_CAP"] = audio_cap    # "320"=画質優先 / "none"=音質優先
    if urls:
        env["DJVM_MANUAL_URL"] = "1"
    else:
        env.pop("DJVM_MANUAL_URL", None)
    try:
        proc = subprocess.Popen([py, str(CORE)], cwd=str(HERE), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace")
        job["proc"] = proc
        proc.stdin.write(answers); proc.stdin.flush(); proc.stdin.close()
        for line in proc.stdout:
            line = line.rstrip("\n"); job["log"].append(line)
            m = re.search(r"✅ 完成:\s*(.+\.mp4)", line)
            if m: job["outputs"].append(m.group(1).strip())
        proc.wait()
    except Exception as e:
        job["log"].append(f"❌ サーバー側エラー: {e}")

def run_job(job_id, items, out_dir, extend, py, audio_cap="320"):
    """items = [(path, url_or_empty), ...]。URLありは URLモード、無しは自動でまとめて実行。"""
    job = JOBS[job_id]
    try:
        auto = [p for (p,u) in items if not u]
        manual = [(p,u) for (p,u) in items if u]
        if auto:
            job["log"].append(f"──── 自動選択で {len(auto)}曲 ────")
            run_engine(job, auto, out_dir, extend, py, audio_cap=audio_cap)
        if manual:
            job["log"].append(f"──── 指定URLで {len(manual)}曲 ────")
            run_engine(job, [p for (p,_) in manual], out_dir, extend, py,
                       urls=[u for (_,u) in manual], audio_cap=audio_cap)
    finally:
        job["done"] = True

PAGE = None  # 下でHTMLを読み込む

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass   # アクセスログ抑制

    def _send(self, code, ctype, body):
        if isinstance(body, str): body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif u.path == "/status":
            q = parse_qs(u.query); jid = q.get("job", [""])[0]
            job = JOBS.get(jid)
            if not job: return self._send(404, "application/json", b'{"error":"no job"}')
            names = self._job_outputs(job)
            data = {"log": job["log"], "done": job["done"], "outputs": names}
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(data, ensure_ascii=False))
        elif u.path == "/download":
            q = parse_qs(u.query); jid = q.get("job",[""])[0]; name = unquote(q.get("name",[""])[0])
            job = JOBS.get(jid)
            if not job: return self._send(404, "text/plain", "no job")
            target = None
            # ①ログから拾ったパス ②出力フォルダ内を実走査（ログ取りこぼし対策）
            for p in job["outputs"]:
                if Path(p).name == name and Path(p).exists(): target = Path(p)
            if target is None and job.get("outdir"):
                cand = Path(job["outdir"]) / name
                if cand.exists(): target = cand
            if not target or not target.exists():
                return self._send(404, "text/plain", "file not found")
            data = target.read_bytes()
            from urllib.parse import quote as _q
            ascii_name = target.name.encode("ascii", "ignore").decode() or "video.mp4"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Disposition",
                             f"attachment; filename=\"{ascii_name}\"; "
                             f"filename*=UTF-8''{_q(target.name)}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        elif u.path == "/reveal":
            # 完成フォルダを Finder で開く（同じMac上で動いているので open が使える）
            q = parse_qs(u.query); jid = q.get("job",[""])[0]
            job = JOBS.get(jid)
            od = job.get("outdir") if job else None
            if od and Path(od).exists():
                try:
                    subprocess.Popen(["open", od])
                    return self._send(200, "text/plain", "ok")
                except Exception as e:
                    return self._send(500, "text/plain", f"error: {e}")
            return self._send(404, "text/plain", "folder not found")
        else:
            self._send(404, "text/plain", "not found")

    def _job_outputs(self, job):
        """完成ファイル名の一覧。ログ由来＋出力フォルダの実走査をマージ（.mp4のみ）。"""
        names = []
        for p in job["outputs"]:
            n = Path(p).name
            if n not in names: names.append(n)
        od = job.get("outdir")
        if od and Path(od).exists():
            for f in sorted(Path(od).glob("*.mp4")):
                if f.name not in names: names.append(f.name)
        return names

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/run":
            ctype = self.headers.get("Content-Type","")
            m = re.search(r'boundary=(.+)$', ctype)
            if not m: return self._send(400,"text/plain","bad form")
            boundary = m.group(1).strip('"').encode()
            length = int(self.headers.get("Content-Length","0"))
            body = self.rfile.read(length)
            files, fields = parse_multipart(body, boundary)
            if not files: return self._send(400,"text/plain","no file")
            jid = uuid.uuid4().hex[:12]
            jdir = WORK / jid; (jdir/"in").mkdir(parents=True, exist_ok=True)
            (jdir/"out").mkdir(parents=True, exist_ok=True)
            music_paths = []
            for idx,(_name, fn, data) in enumerate(files):
                safe = re.sub(r'[/\\]', "_", fn)
                fp = jdir/"in"/safe; fp.write_bytes(data); music_paths.append(fp)
            # 各曲のURL（url_0, url_1 ... 空なら自動選択）
            items = []
            for idx, p in enumerate(music_paths):
                u = fields.get(f"url_{idx}", "").strip()
                if u:
                    clean = normalize_youtube_url(u)
                    u = clean if clean else ""   # YouTube URLとして解釈できなければ自動選択
                items.append((p, u))
            extend = fields.get("extend","") == "1"
            # 音声の扱い："320"=画質優先(既定) / "none"=音質優先
            audio_cap = "none" if fields.get("audio_cap","320") == "none" else "320"
            browser = fields.get("browser","chrome")
            set_browser(browser)
            JOBS[jid] = {"log":[], "done":False, "outputs":[], "proc":None, "outdir": str(jdir/"out")}
            py = os.environ.get("DJVM_PYTHON", sys.executable)
            threading.Thread(target=run_job,
                args=(jid, items, jdir/"out", extend, py, audio_cap),
                daemon=True).start()
            self._send(200,"application/json",
                       json.dumps({"job":jid}).encode())
        elif u.path == "/repair":
            jid = uuid.uuid4().hex[:12]
            JOBS[jid] = {"log":[], "done":False, "outputs":[], "proc":None}
            py = os.environ.get("DJVM_PYTHON", sys.executable)
            threading.Thread(target=run_repair, args=(jid, py), daemon=True).start()
            self._send(200,"application/json", json.dumps({"job":jid}).encode())
        else:
            self._send(404,"text/plain","not found")

def main():
    global PAGE, PORT
    ensure_config()
    html = HERE / "web_ui.html"
    PAGE = html.read_text(encoding="utf-8") if html.exists() else "<h1>web_ui.html がありません</h1>"
    socketserver.TCPServer.allow_reuse_address = True
    # 既定ポートが使用中なら、空いているポートを自動で探す（前のサーバーが残っていても起動できる）
    httpd = None
    for p in [PORT, PORT+1, PORT+2, PORT+3, PORT+4, 0]:
        try:
            httpd = socketserver.ThreadingTCPServer(("127.0.0.1", p), Handler)
            PORT = httpd.server_address[1]
            break
        except OSError:
            continue
    if httpd is None:
        print("❌ 空きポートが見つかりませんでした。既存のサーバーを終了して再実行してください。")
        return
    # 実際に使うポートをファイルに書く（起動スクリプトがブラウザを開く先に使う）
    try:
        (CONFIG_DIR / "web_port").write_text(str(PORT))
    except Exception:
        pass
    with httpd:
        print(f"✅ DJ Video Maker Web はこちら → http://127.0.0.1:{PORT}")
        print("   （このウィンドウは開いたままにしてください。閉じると終了します）")
        try: httpd.serve_forever()
        except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
