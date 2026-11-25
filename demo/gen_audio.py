import json, urllib.request, urllib.parse, pathlib
api = "http://127.0.0.1:8020"
TEXT = "Hello, people who dont show off are equally skilled as people who shit post about everything everyday"
info = json.load(urllib.request.urlopen(api + "/api/info"))
pick = {"en": ("kokoro", "af_heart"), "de": ("piper", None), "fr": ("kokoro", None), "it": ("kokoro", None), "es": ("kokoro", None), "hi": ("kokoro", None), "zh": ("kokoro", None)}
out = {}
for code, (eng, voice) in pick.items():
    lang = next(l for l in info["languages"] if l["code"] == code); e = next(x for x in lang["engines"] if x["id"] == eng)
    v = voice or e["voices"][0]["id"]
    body = json.dumps({"text": TEXT, "language": code, "engine": eng, "voice": v, "speed": 1.0, "translate": code != "en"}).encode()
    req = urllib.request.Request(api + "/api/tts", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        audio = r.read(); spoken = urllib.parse.unquote(r.headers.get("X-Spoken-Text", "")); secs = r.headers.get("X-Audio-Seconds")
    pathlib.Path(f"assets/{code}.wav").write_bytes(audio); out[code] = {"name": lang["name"], "engine": eng, "voice": v, "spoken": spoken, "seconds": float(secs)}
    print(code, eng, v, secs, "s |", spoken, flush=True)
json.dump({"input": TEXT, "langs": out}, open("assets/material.json", "w"), ensure_ascii=False, indent=1)
