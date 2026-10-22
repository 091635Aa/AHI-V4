# -*- coding: utf-8 -*-
"""meeting59 / refs-web：B 站 API 工具（wbi 签名，仅标准库；不做任何转码）
用法:
  python 工具_bili.py search <关键词> [页]
  python 工具_bili.py view <BV号>
  python 工具_bili.py playurl <BV号> <cid> [输出.m4s]
"""
import sys, json, time, re, hashlib, urllib.request, urllib.parse, http.cookiejar

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
MIXIN = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
         33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
         61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62,
         11, 36, 20, 34, 44, 52]

_cj = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cj))


def _get(url, referer="https://www.bilibili.com", timeout=25, raw=False):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Referer": referer,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    with _opener.open(req, timeout=timeout) as r:
        data = r.read()
    return data if raw else json.loads(data.decode("utf-8", "replace"))


def _warm():
    try:
        _get("https://www.bilibili.com/", timeout=20, raw=True)
    except Exception as e:
        print("[warn] warm cookie:", e, file=sys.stderr)


def wbi_keys():
    d = _get("https://api.bilibili.com/x/web-interface/nav")
    img = d["data"]["wbi_img"]["img_url"].rsplit("/", 1)[-1].split(".")[0]
    sub = d["data"]["wbi_img"]["sub_url"].rsplit("/", 1)[-1].split(".")[0]
    raw = img + sub
    return "".join(raw[i] for i in MIXIN)[:32]


def signed(params, referer="https://www.bilibili.com"):
    key = wbi_keys()
    p = dict(params)
    p["wts"] = int(time.time())
    items = sorted(p.items())
    q = urllib.parse.urlencode([(k, re.sub(r"[!'()*]", "", str(v))) for k, v in items])
    p["w_rid"] = hashlib.md5((q + key).encode()).hexdigest()
    return urllib.parse.urlencode(sorted(p.items()))


def cmd_search(kw, page=1):
    q = signed({"search_type": "video", "keyword": kw, "page": page})
    d = _get("https://api.bilibili.com/x/web-interface/wbi/search/type?" + q,
             referer="https://search.bilibili.com/")
    print("code=%s message=%s" % (d.get("code"), d.get("message")))
    res = d.get("data", {}).get("result") or []
    for it in res:
        title = re.sub(r"<[^>]+>", "", it.get("title", ""))
        print("%-12s %-9s dur=%-6s play=%-9s up=%-14s %s" % (
            it.get("bvid"), it.get("pubdate"),
            it.get("duration"), it.get("play"), it.get("author"), title))


def cmd_view(bv):
    d = _get("https://api.bilibili.com/x/web-interface/view?bvid=" + bv)
    if d.get("code") != 0:
        print(json.dumps(d, ensure_ascii=False)); return
    v = d["data"]
    print(json.dumps({
        "bvid": v["bvid"], "aid": v["aid"], "title": v["title"],
        "owner": v["owner"]["name"], "owner_mid": v["owner"]["mid"],
        "pubdate": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["pubdate"])),
        "duration": v["duration"], "desc": v["desc"], "copyright": v["copyright"],
        "pages": [{"cid": p["cid"], "part": p["part"], "duration": p["duration"]} for p in v["pages"]],
        "dimension": v.get("dimension"),
    }, ensure_ascii=False, indent=1))


def cmd_playurl(bv, cid, out=None):
    d = _get("https://api.bilibili.com/x/web-interface/view?bvid=" + bv)
    aid = d["data"]["aid"]
    q = signed({"avid": aid, "cid": cid, "fnval": 16, "fnver": 0, "fourk": 0,
                "otype": "json", "qn": 0, "platform": "pc"})
    p = _get("https://api.bilibili.com/x/player/wbi/playurl?" + q)
    print("code=%s message=%s" % (p.get("code"), p.get("message")))
    if p.get("code") != 0:
        print(json.dumps(p, ensure_ascii=False)[:800]); return
    dash = p["data"].get("dash")
    if not dash:
        print("no dash; durl:", json.dumps(p["data"].get("durl"), ensure_ascii=False)[:300]); return
    auds = dash["audio"]
    # 选择码率最高的普通音频流
    best = max(auds, key=lambda a: a.get("bandwidth", 0))
    print("[audio] id=%s codec=%s bandwidth=%s baseUrl(host)=%s" % (
        best["id"], best.get("codecs"), best["bandwidth"],
        best["baseUrl"].split("/")[2]))
    if out:
        data = _get(best["baseUrl"], raw=True, referer="https://www.bilibili.com")
        with open(out, "wb") as f:
            f.write(data)
        print("[saved] %s (%d bytes)" % (out, len(data)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        cmd_search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    elif cmd == "view":
        cmd_view(sys.argv[2])
    elif cmd == "playurl":
        cmd_playurl(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
