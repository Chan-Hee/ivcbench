#!/usr/bin/env python3
"""Range-fetch ONLY the 10x triplet + metadata from BioStudies S-BSST2978,
skipping the 7.6 GB Seurat .rds. Total download ~438 MB.
Verified 2026-08-31: MTX = 20953 genes x 43112 cells, 143,951,859 nnz;
metadata has donor.id (D1-D4) x cytokine.condition (UNS/Th0/Th2/Th17/iTreg)."""
import subprocess, struct, zlib, os, sys

URL = ("https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/978/S-BSST2978/"
       "Files/scRNAseq.zip")
CD_OFF, CD_SIZE = 8027319652, 1402          # ZIP64 central directory
OUT = sys.argv[1] if len(sys.argv) > 1 else "./canogamez2020"
WANT = ("metadata.txt", "genes.tsv", "barcodes.tsv", "raw_UMIs.mtx")

def rng(a, b):
    return subprocess.run(["curl", "-sS", "--retry", "5", "-r", f"{a}-{b}", URL],
                          capture_output=True, check=True).stdout

def central_dir():
    cd, p, out = rng(CD_OFF, CD_OFF + CD_SIZE - 1), 0, {}
    while p < len(cd) - 4 and cd[p:p+4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p+10:p+12])[0]
        csz, usz = struct.unpack("<II", cd[p+20:p+28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p+28:p+34])
        lho = struct.unpack("<I", cd[p+42:p+46])[0]
        name = cd[p+46:p+46+nlen].decode()
        extra, q = cd[p+46+nlen:p+46+nlen+elen], 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q+4]); d, k = extra[q+4:q+4+hsz], 0
            if hid == 1:
                if usz == 0xFFFFFFFF: usz = struct.unpack("<Q", d[k:k+8])[0]; k += 8
                if csz == 0xFFFFFFFF: csz = struct.unpack("<Q", d[k:k+8])[0]; k += 8
                if lho == 0xFFFFFFFF: lho = struct.unpack("<Q", d[k:k+8])[0]; k += 8
            q += 4 + hsz
        out[name] = (lho, csz, usz, method)
        p += 46 + nlen + elen + clen
    return out

os.makedirs(OUT, exist_ok=True)
for name, (lho, csz, usz, method) in central_dir().items():
    if "__MACOSX" in name or not name.endswith(WANT):
        continue
    lh = rng(lho, lho + 29)
    nl, el = struct.unpack("<HH", lh[26:30])
    start = lho + 30 + nl + el
    dst = os.path.join(OUT, os.path.basename(name))
    print(f"-> {dst}  ({csz/1e6:.1f} MB compressed / {usz/1e6:.1f} MB raw)", flush=True)
    dec = zlib.decompressobj(-15)
    with open(dst, "wb") as fh:
        pos, CHUNK = start, 32 << 20
        while pos <= start + csz - 1:
            end = min(pos + CHUNK - 1, start + csz - 1)
            fh.write(dec.decompress(rng(pos, end)))
            pos = end + 1
        fh.write(dec.flush())
print("done ->", OUT)
