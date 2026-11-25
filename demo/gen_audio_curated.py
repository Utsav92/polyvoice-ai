import json, urllib.request, pathlib
api = "http://127.0.0.1:8020"
INPUT = "Hello, people who dont show off are equally skilled as people who shit post about everything everyday"
CUR = {  # hand-written translations of the input (the machine output mangles the slang; see material_raw.json)
 "en": INPUT,
 "de": "Hallo, Leute, die nicht angeben, sind genauso talentiert wie Leute, die jeden Tag über alles Unsinn posten.",
 "fr": "Bonjour, les gens qui ne se vantent pas sont aussi doués que ceux qui postent n'importe quoi sur tout, tous les jours.",
 "it": "Ciao, le persone che non si mettono in mostra sono brave quanto quelle che postano sciocchezze su tutto, ogni giorno.",
 "es": "Hola, las personas que no presumen son igual de hábiles que las que publican tonterías sobre todo, todos los días.",
 "hi": "नमस्ते, जो लोग दिखावा नहीं करते, वे उतने ही काबिल होते हैं जितने वे लोग जो हर रोज़ हर चीज़ के बारे में बकवास पोस्ट करते हैं।",
 "zh": "大家好，不炫耀的人和每天对什么都乱发帖的人一样有本事。"}
info = json.load(urllib.request.urlopen(api + "/api/info")); raw = json.load(open("assets/material_raw.json"))["langs"]; out = {}
for code, text in CUR.items():
    r_ = raw[code]; body = json.dumps({"text": text, "language": code, "engine": r_["engine"], "voice": r_["voice"], "speed": 1.0, "translate": False}).encode()
    with urllib.request.urlopen(urllib.request.Request(api + "/api/tts", data=body, headers={"Content-Type": "application/json"}), timeout=600) as r:
        pathlib.Path(f"assets/{code}.wav").write_bytes(r.read()); secs = float(r.headers["X-Audio-Seconds"])
    out[code] = {"name": r_["name"], "engine": r_["engine"], "voice": r_["voice"], "spoken": text, "seconds": secs}; print(code, secs, "s")
json.dump({"input": INPUT, "langs": out}, open("assets/material.json", "w"), ensure_ascii=False, indent=1)
