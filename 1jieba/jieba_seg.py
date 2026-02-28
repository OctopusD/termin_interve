import jieba
from tqdm import tqdm
import json
import re
from itertools import islice
from sacremoses import MosesTokenizer

IN = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/other/data/filter.jsonl"
OUT_JSONL = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/termin_interve/jieba_seg/data/en-zh_tok.jsonl"
OUT_FA = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/termin_interve/jieba_seg/data/en-zh.fastalign.txt"


mtok = MosesTokenizer(lang="en")

# 保护占位符：%1、%2、%1!s!、%2!d! 等
PH_PAT = re.compile(r"%\d+(?:![A-Za-z]+!)?")

def clean_en(s: str) -> str:
    s = s.replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_zh(s: str) -> str:
    s = s.replace('"""', '').replace('""', '').replace('"', '')
    s = s.replace("\\r\\n", " ").replace("\\n", " ").replace("\\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def en_moses_tokenize(s: str) -> str:
    """
    Moses 英文分词 + 占位符保护
    """
    s = clean_en(s)

    placeholders = []
    def _ph_repl(m):
        placeholders.append(m.group(0))
        return f"PLACEHOLDER{len(placeholders)-1}"

    # 先保护占位符
    s = PH_PAT.sub(_ph_repl, s)

    toks = mtok.tokenize(s, return_str=False)

    restored = []
    for t in toks:
        m = re.fullmatch(r"PLACEHOLDER(\d+)", t)
        restored.append(placeholders[int(m.group(1))] if m else t)

    # 过滤纯符号 token（保守）
    restored = [t for t in restored if not re.fullmatch(r"\W+", t)]

    return " ".join(restored)

def jieba_seg_zh(s: str) -> str:
    s = clean_zh(s)
    toks = list(jieba.cut(s, cut_all=False))
    toks = [t for t in toks if not re.fullmatch(r"\W+", t)]
    return " ".join(toks)

with open(IN, "r", encoding="utf-8") as fin, \
     open(OUT_JSONL, "w", encoding="utf-8") as fjson, \
     open(OUT_FA, "w", encoding="utf-8") as ffa:

    for line in tqdm(fin, desc="tokenize(moses+jieba)", unit="line"):
        obj = json.loads(line)

        en = en_moses_tokenize(obj["inputs"])
        zh = jieba_seg_zh(obj["targets"])

        fjson.write(json.dumps({"en": en, "zh": zh}, ensure_ascii=False) + "\n")
        ffa.write(en + " ||| " + zh + "\n")

print("done")
print("jsonl:", OUT_JSONL)
print("fast_align input:", OUT_FA)
