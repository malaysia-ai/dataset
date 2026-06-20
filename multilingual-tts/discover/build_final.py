#!/usr/bin/env python3
"""Join per-repo verification results with candidate metadata and emit the final
deliverable: non-overlapping, verified-suitable TTS/STT datasets not already in
Multilingual-TTS. Sorted by trending score then downloads."""
import os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
VDIR = os.path.join(CACHE, "verify")

meta = {d["id"]: d for d in json.load(open(f"{CACHE}/candidates_strong.json"))}

def base(v, m):
    return {
        "id": v["id"],
        "author": m.get("author"),
        "trendingScore": m.get("trendingScore", 0),
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "language": m.get("language", []),
        "task_categories": m.get("task_categories", []),
        "url": f"https://huggingface.co/datasets/{v['id']}",
    }

GATED = ("gatedrepo", "client error", "401", "403")
rows, likely, gated, no_parquet = [], [], [], []
counts = {}
for fn in os.listdir(VDIR):
    if not fn.endswith(".json"):
        continue
    try:
        v = json.load(open(os.path.join(VDIR, fn)))
    except Exception:
        continue
    verdict = v.get("verdict") or "null"
    counts[verdict] = counts.get(verdict, 0) + 1
    m = meta.get(v["id"], {})
    err = (v.get("error") or "").lower()
    if verdict == "suitable":
        r = base(v, m)
        r.update({"audio_col": v.get("audio_col"), "text_col": v.get("text_col"),
                  "samplerate": v.get("samplerate"), "sample_text": v.get("sample_text"),
                  "n_parquet": v.get("n_parquet"), "verified_shard": v.get("shard")})
        rows.append(r)
    elif (verdict == "not-suitable" and v.get("audio_col") and v.get("text_col")
          and v.get("sample_text") and not v.get("audio_decoded")
          and not v.get("neg_domain_hit")):
        # audio column + real transcription present, but audio stored as a path
        # reference (not embedded bytes) so it couldn't be byte-verified here
        r = base(v, m)
        r.update({"audio_col": v.get("audio_col"), "text_col": v.get("text_col"),
                  "sample_text": v.get("sample_text"), "verified_shard": v.get("shard"),
                  "note": "audio column + transcription present; audio not byte-decodable "
                          "from row group 0 (likely path-referenced files) -- confirm manually"})
        likely.append(r)
    elif verdict == "error" and any(g in err for g in GATED):
        gated.append(base(v, m))
    elif verdict == "needs-manual":
        r = base(v, m); r["other_formats"] = v.get("other_formats"); no_parquet.append(r)

sk = lambda r: (-(r["trendingScore"] or 0), -(r["downloads"] or 0), -(r["likes"] or 0))
rows.sort(key=sk); likely.sort(key=sk); gated.sort(key=sk); no_parquet.sort(key=sk)

out = {
    "description": "New HF audio datasets suitable for TTS/STT, NOT already in "
                   "malaysia-ai/Multilingual-TTS. Verified per-repo by reading the "
                   "README + smallest parquet shard (row group 0): confirmed an "
                   "audio column + transcription column, decoded one audio sample, "
                   "and no environment/animal/music signal in the README.",
    "source": "https://huggingface.co/datasets?modality=modality:audio&sort=trending",
    "candidate_pool": "strong (declared ASR/TTS/text-to-audio task category), "
                      "overlap with Multilingual-TTS removed",
    "verdict_counts": counts,
    "total_suitable": len(rows),
    "total_likely_suitable": len(likely),
    "total_gated_needs_license": len(gated),
    "total_no_parquet_needs_manual": len(no_parquet),
    "datasets": rows,
    "likely_suitable_audio_not_byte_verified": likely,
    "gated_candidates_need_license_acceptance": gated,
    "no_parquet_candidates_need_manual_check": no_parquet,
}
with open(f"{ROOT}/new_tts_stt_datasets.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("verdict counts:", counts)
print("verified suitable:", len(rows))
print("written:", f"{ROOT}/new_tts_stt_datasets.json")
print("\nTop 20 suitable:")
for r in rows[:20]:
    print(f"  ts={r['trendingScore']:>2} dl={r['downloads']:>8} | {r['id']}  [{r['audio_col']}/{r['text_col']}]")
