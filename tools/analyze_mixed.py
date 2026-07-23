import os

BASE = "data/newbatch/mixed_probe"
SRC = os.path.join(BASE, "01-RAW-Photos")
EDT = os.path.join(BASE, "1502 - 98 images")

def names(d):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg")))

src = names(SRC)
edt = names(EDT)
print(f"source (before): {len(src)}  | edited (after): {len(edt)}")
print("source sample:", src[:4])
print("edited sample:", edt[:4])

sset, eset = set(src), set(edt)
exact = sset & eset
print(f"\nEXACT filename overlap: {len(exact)} / {len(edt)} edited")
print("sample matched:", sorted(exact)[:4])
print("edited NOT in source:", sorted(eset - sset)[:6])

# maybe edited names are a prefix/substring of source (bracket base)
if len(exact) < len(edt) * 0.5:
    # try: does each edited name appear as substring of some source name?
    sub = 0
    for e in edt:
        if any(e in s or s in e for s in src):
            sub += 1
    print(f"\nsubstring-match fallback: {sub}/{len(edt)}")
