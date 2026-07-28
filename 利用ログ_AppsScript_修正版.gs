// ============================================================
//  DJ Video Maker 利用ログ
//    log  … 1リクエスト＝1行の生ログ
//    集計 … 人ごとの利用状況
//
//  記録するのは「起動したか」「何曲成功/失敗したか」「所要時間」だけで、
//  曲名やファイル名は送っていない（利用者の音楽の中身は残さない）。
// ============================================================

var LOG_HEADER = ['日時', 'ユーザー名', '端末名', 'OS', 'バージョン',
                  '種別', '成功曲数', '失敗曲数', '所要秒'];


function doGet(e) {
  try {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var raw = ss.getSheetByName('log');
    if (!raw) {
      raw = ss.insertSheet('log');
      raw.appendRow(LOG_HEADER);
    } else if (raw.getLastColumn() < LOG_HEADER.length) {
      // 旧フォーマット(5列)のシートに、新しい見出しを足す
      raw.getRange(1, 1, 1, LOG_HEADER.length).setValues([LOG_HEADER]);
    }

    var p = (e && e.parameter) ? e.parameter : {};
    var ts = p.ts || Utilities.formatDate(
        new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss');

    raw.appendRow([
      ts,
      p.user || '(unknown)',
      p.host || '',
      p.os   || '',
      p.ver  || '',
      p.ev   || 'launch',      // launch=起動 / done=動画作成おわり
      p.ok   || '',
      p.ng   || '',
      p.sec  || ''
    ]);

    rebuildSummary(ss);
    return ContentService.createTextOutput('ok');
  } catch (err) {
    return ContentService.createTextOutput('err: ' + err);
  }
}


/**
 * ターミナル出力の全文を受け取る（1回の作成＝1行）。
 *
 * 量が多いので「詳細ログ」という別シートに入れる。log/集計 と混ぜると
 * 集計タブが重くなって読めなくなるため。
 *
 * ★このログには曲名・ファイルパス・YouTubeのURLがそのまま入る。
 *   配布時に、記録している旨を必ず利用者へ伝えること。
 */
function doPost(e) {
  try {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sh = ss.getSheetByName('詳細ログ');
    if (!sh) {
      sh = ss.insertSheet('詳細ログ');
      sh.appendRow(['日時', 'ユーザー名', '端末名', 'OS', 'バージョン', '全文']);
      sh.setColumnWidth(6, 600);
    }
    var p = (e && e.parameter) ? e.parameter : {};
    var ts = p.ts || Utilities.formatDate(
        new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss');
    var text = String(p.text || '');
    if (text.length > 49000) text = text.slice(0, 49000) + '\n…（長すぎるため省略）';

    sh.appendRow([ts, p.user || '(unknown)', p.host || '', p.os || '',
                  p.ver || '', text]);
    // 新しい行を折り返さず1行表示にして、シートを見やすく保つ
    sh.getRange(sh.getLastRow(), 6).setWrap(false);
    return ContentService.createTextOutput('ok');
  } catch (err) {
    return ContentService.createTextOutput('err: ' + err);
  }
}


/**
 * 日時を「比較できる数値」に直す。
 *
 * ★ここが以前のバグの原因。
 *   appendRow で 'yyyy-MM-dd HH:mm:ss' の文字列を書いても、スプレッドシートが
 *   日付型に変換して保存する。読み戻すと Date オブジェクトになり、
 *   String(Date) は "Sat Jul 25 2026 17:59:00 GMT+0900" という形式になる。
 *   先頭が曜日名なので、文字列比較すると日付順ではなく曜日のアルファベット順
 *   （Fri < Mon < Sat < Sun < Thu < Tue < Wed）で比べられてしまい、
 *   「最終利用・端末名・バージョンがずっと変わらない」状態になっていた。
 */
function toMillis(v) {
  if (v instanceof Date) return v.getTime();
  var s = String(v || '').trim();
  if (!s) return 0;
  var d = new Date(s.replace(' ', 'T'));   // 'yyyy-MM-dd HH:mm:ss' を ISO 形式に
  if (!isNaN(d.getTime())) return d.getTime();
  d = new Date(s);
  return isNaN(d.getTime()) ? 0 : d.getTime();
}


function toText(v) {
  var ms = toMillis(v);
  if (!ms) return String(v || '');
  return Utilities.formatDate(new Date(ms), Session.getScriptTimeZone(),
                              'yyyy-MM-dd HH:mm:ss');
}


function num(v) {
  var n = parseInt(String(v || '').trim(), 10);
  return isNaN(n) ? 0 : n;
}


/**
 * 集計タブだけを今すぐ作り直す（動作確認用）。
 *
 * rebuildSummary は doGet からしか呼ばれないため、コードを直しても
 * 新しいアクセスが来るまで集計タブは古いままになる。
 * Apps Scriptエディタでこの関数を選んで「実行」すれば、
 * デプロイしなくてもその場で作り直せる。
 */
function 集計を作り直す() {
  rebuildSummary(SpreadsheetApp.getActiveSpreadsheet());
}


function rebuildSummary(ss) {
  var raw = ss.getSheetByName('log');
  if (!raw) return;
  var data = raw.getDataRange().getValues();
  if (data.length < 2) return;

  var byUser = {};
  for (var i = 1; i < data.length; i++) {
    var row  = data[i];
    var ts   = row[0], user = row[1], host = row[2], ver = row[4];
    var ev   = String(row[5] || 'launch').trim() || 'launch';  // 旧行は起動扱い
    if (!user) continue;

    if (!byUser[user]) {
      byUser[user] = { lastMs: -1, last: ts, host: host, ver: ver,
                       launches: 0, runs: 0, ok: 0, ng: 0, sec: 0 };
    }
    var u = byUser[user];
    if (ev === 'done') {
      u.runs += 1;
      u.ok   += num(row[6]);
      u.ng   += num(row[7]);
      u.sec  += num(row[8]);
    } else {
      u.launches += 1;
    }
    // ★数値（ミリ秒）で比較する。これで本当に新しい記録だけが採用される。
    var ms = toMillis(ts);
    if (ms >= u.lastMs) {
      u.lastMs = ms; u.last = ts;
      if (host) u.host = host;
      if (ver)  u.ver  = ver;
    }
  }

  var sum = ss.getSheetByName('集計');
  if (!sum) sum = ss.insertSheet('集計');
  sum.clear();
  sum.appendRow(['ユーザー名', '端末名', '最終利用', '起動回数', '作成回数',
                 '成功曲数', '失敗曲数', '平均秒/回', 'バージョン']);

  var users = Object.keys(byUser).sort(function (a, b) {
    return byUser[b].lastMs - byUser[a].lastMs;   // 最近使った人が上
  });
  users.forEach(function (name) {
    var u = byUser[name];
    var avg = u.runs > 0 ? Math.round(u.sec / u.runs) : '';
    sum.appendRow([name, u.host, toText(u.last), u.launches, u.runs,
                   u.ok, u.ng, avg, u.ver]);
  });
}
