import json
from collections import Counter
from probe_samplerate import write_markdown
res={}
for l in open("out/sr_probe_results.jsonl"):
    try: r=json.loads(l)
    except: continue
    res[r["id"]]=r  # dedup: keep last
res=list(res.values())
write_markdown(res, 44000, "out/datasets_ge_44k.md", len(res))
ge=[r for r in res if isinstance(r.get("mode_sr"),int) and r["mode_sr"]>=44000]
byb=Counter(r.get("bucket") for r in ge)
summary={"unique_probed":len(res),"ge_44k":len(ge),
         "decoded_ok":sum(1 for r in res if r.get("mode_sr")),
         "errors":sum(1 for r in res if r.get("error")),
         "gated_blocked":sum(1 for r in res if str(r.get("error","")).startswith("gated")),
         "timeouts":sum(1 for r in res if r.get("error")=="timeout"),
         "ge44k_by_bucket":dict(byb),
         "sr_hist":{str(k):v for k,v in Counter(r["mode_sr"] for r in res if isinstance(r.get("mode_sr"),int)).most_common()}}
json.dump(summary, open("out/sr_summary.json","w"), indent=2)
print(json.dumps(summary, indent=2))
