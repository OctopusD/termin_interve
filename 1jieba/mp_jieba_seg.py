import re
import jieba
from tqdm import tqdm
from sacremoses import MosesTokenizer
from multiprocessing import Pool, cpu_count

IN = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/filter/data/filter.jsonl"
OUT_JSONL = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/termin_interve/jieba_seg/data/en-zh_tok.jsonl"
OUT_FA = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/termin_interve/jieba_seg/data/en-zh.fastalign.txt"

# 可选：更快 JSON
try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    import json
    _USE_ORJSON = False

PH_PAT = re.compile(r"%\d+(?:![A-Za-z]+!)?")
WS_PAT = re.compile(r"\s+")
SYM_PAT = re.compile(r"\W+")
PH_TOK_PAT = re.compile(r"^PLACEHOLDER(\d+)$")

# 每个进程各自持有 tokenizer
_mtok = None

def _init_worker():
    global _mtok
    _mtok = MosesTokenizer(lang="en")
    # jieba 在 worker 中首次使用会加载词典；提前触发可减少抖动（可选）
    jieba.initialize()

def clean_en(s: str) -> str:
    s = s.replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ")
    return WS_PAT.sub(" ", s).strip()

def clean_zh(s: str) -> str:
    s = s.replace('"""', '').replace('""', '').replace('"', '')
    s = s.replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ")
    return WS_PAT.sub(" ", s).strip()

def en_moses_tokenize(s: str) -> str:
    global _mtok
    s = clean_en(s)

    placeholders = []
    def _ph_repl(m):
        placeholders.append(m.group(0))
        return f"PLACEHOLDER{len(placeholders)-1}"

    s = PH_PAT.sub(_ph_repl, s)

    toks = _mtok.tokenize(s, return_str=False)

    restored = []
    for t in toks:
        m = PH_TOK_PAT.match(t)
        if m:
            restored.append(placeholders[int(m.group(1))])
        else:
            restored.append(t)

    # 过滤纯符号 token（保守）
    restored = [t for t in restored if not SYM_PAT.fullmatch(t)]
    return " ".join(restored)

def jieba_seg_zh(s: str) -> str:
    s = clean_zh(s)
    toks = jieba.cut(s, cut_all=False)
    toks = [t for t in toks if not SYM_PAT.fullmatch(t)]
    return " ".join(toks)

def _process_line(line: bytes):
    # line: bytes（更快）
    if _USE_ORJSON:
        obj = orjson.loads(line)
        en = en_moses_tokenize(obj["inputs"])
        zh = jieba_seg_zh(obj["targets"])
        out_json = orjson.dumps({"en": en, "zh": zh}) + b"\n"
    else:
        line_str = line.decode("utf-8")
        obj = json.loads(line_str)
        en = en_moses_tokenize(obj["inputs"])
        zh = jieba_seg_zh(obj["targets"])
        out_json = (json.dumps({"en": en, "zh": zh}, ensure_ascii=False) + "\n").encode("utf-8")

    out_fa = (en + " ||| " + zh + "\n").encode("utf-8")
    return out_json, out_fa

def main():
    nproc = max(1, min(cpu_count(), 16))  # 你可以按机器核数调
    chunksize = 256  # 通常 128~1024 都可以试

    with open(IN, "rb") as fin, open(OUT_JSONL, "wb") as fjson, open(OUT_FA, "wb") as ffa:
        with Pool(processes=nproc, initializer=_init_worker) as pool:
            it = pool.imap(_process_line, fin, chunksize=chunksize)
            for out_json, out_fa in tqdm(it, desc=f"tokenize({nproc} procs)", unit="line"):
                fjson.write(out_json)
                ffa.write(out_fa)

    print("done")
    print("jsonl:", OUT_JSONL)
    print("fast_align input:", OUT_FA)

if __name__ == "__main__":
    main()
