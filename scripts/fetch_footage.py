#!/usr/bin/env python3
"""Download the demo corpus from Wikimedia Commons.

Every file here is CC BY or CC BY-SA, which means it is safe to show publicly with
attribution - unlike broadcast football, which is why the usual football datasets
(SoccerNet and friends) cannot appear in a demo video.

Run this on whichever machine will do the processing; the VM can reach Commons directly.
"""
import json, os, sys, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UA = "FredLasso-hackathon/0.1 (VAST Builders Challenge)"
RAW = "data/raw"

MANUAL = [
    ("utrecht_ajax_mar2020.webm",
     "https://upload.wikimedia.org/wikipedia/commons/5/52/Utrecht_v_Ajax_Mar_2020.webm",
     "CC BY 3.0", "FC Utrecht v Ajax, KNVB Cup SF, 4 Mar 2020 (2-0). Author: FCUFAN."),
    ("utrecht_ajax_dec2018.webm",
     "https://upload.wikimedia.org/wikipedia/commons/a/af/Utrecht_v_Ajax_Dec_2018.webm",
     "CC BY 3.0", "FC Utrecht v Ajax, Eredivisie, Dec 2018 (1-3)."),
]


def commons_search(query, limit=30):
    """Resolve the U-17 clip set by search so the URLs cannot go stale."""
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json"
           "&generator=search&gsrnamespace=6&gsrlimit=%d&gsrsearch=%s"
           "&prop=imageinfo&iiprop=url|size|extmetadata" % (limit, urllib.parse.quote(query)))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    out = []
    for page in data.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {}) or {}
        out.append({
            "title": page["title"][5:],
            "url": info.get("url", "").split("?")[0],
            "size": info.get("size", 0),
            "licence": (meta.get("LicenseShortName", {}) or {}).get("value", "?"),
        })
    return sorted(out, key=lambda r: r["title"])


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have  {dest}")
        return dest
    print(f"  get   {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=900) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return dest


def main():
    os.makedirs(RAW, exist_ok=True)
    manifest = []

    print("U-17 Women's World Cup 2018, New Zealand v Canada (CC BY-SA 4.0):")
    hits = commons_search("incategory:\"Videos of association football\" "
                          "2018 FIFA U-17 Women's World Cup New Zealand Canada")
    for index, hit in enumerate([h for h in hits if "U-17" in h["title"]], 1):
        dest = os.path.join(RAW, f"u17_nzl_can_{index:02d}.webm")
        download(hit["url"], dest)
        manifest.append({"file": dest, "licence": hit["licence"], "source": hit["title"]})

    print("Full matches (CC BY 3.0):")
    for name, url, licence, note in MANUAL:
        dest = os.path.join(RAW, name)
        try:
            download(url, dest)
            manifest.append({"file": dest, "licence": licence, "source": note})
        except Exception as exc:  # noqa: BLE001
            print(f"  skip  {name}: {exc}")

    os.makedirs("data/out", exist_ok=True)
    with open("data/out/footage_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{len(manifest)} files in {RAW}. Attribution recorded in data/out/footage_manifest.json")
    print("Next: python scripts/build_match.py  (stitches the U-17 clips into one match)")


if __name__ == "__main__":
    main()
