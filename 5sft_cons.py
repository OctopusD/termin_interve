#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

# =========================
# 配置区：只改这里
# =========================

# 1) 术语候选（白名单）：intervention_candidates.tsv
CAND_TSV = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/termin_interve/interve_candid4/data/intervention_candidates.tsv"

# 2) 原训练数据（回扫）
TRAIN_JSONL = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/filter/data/filter.jsonl"
SRC_FIELD = "inputs"
TGT_FIELD = "targets"

# 3) 输出：SFT 种子数据（不改写）
OUT_JSONL = "/data/vjuicefs_ai_translation_wl/public_data/11189869/bussi_capability/termin_interve/sft_seed5/data/sft_seed_raw.jsonl"

# terms 的中文指定译法来自 TSV 的哪一列：zh1 或 zh2
USE_ZH_COL = "zh2"   # 推荐 zh2 作为“干预目标”；想要默认译法就改 zh1

# 每条样本最多选几个 term
MAX_TERMS_PER_SAMPLE = 2

# 每个 term 最多保留多少条样本（防止头部术语刷屏）
MAX_SAMPLES_PER_TERM = 200

# 全局最多输出多少条（0 表示不限制）
MAX_OUT = 0

# 试跑：只扫前 N 行训练数据；0 表示全量
MAX_SCAN_LINES = 0

# =========================
# 兜底过滤（理论上 TSV 已经干净，但加一层保险）
# =========================

EN_STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","as","by","at","from",
    "is","are","was","were","be","been","being","it","its","this","that","these","those",
    "not","no","but","if","then","than","which","who","whom","what","when","where","why","how",
    "can","could","may","might","must","shall","should","will","would","do","does","did",
    "also","such","other","any","all","each","both","either","neither","more","most","less","least"
}

# “简单词”：只包含字母/数字/下划线（可用 token 交集快速命中）
RE_SIMPLE = re.compile(r"^[A-Za-z0-9_]+$")

# 英文 token 化：取 word token（含数字/下划线），全部 lower
RE_WORD_TOK = re.compile(r"[A-Za-z0-9_]+")

# =========================
# 读取 TSV 候选池
# =========================

def load_tsv_candidates(path: str, use_zh_col: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    返回：
      en2zh: en_lower -> zh_spec
      en2prio: en_lower -> priority(score)，用于同一句中命中多个时优先选谁
    TSV 期望列名包含：en, score, zh1, zh2（你现在的文件就是这样）
    """
    en2zh: Dict[str, str] = {}
    en2prio: Dict[str, float] = {}

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}

        # 必要列
        if "en" not in col:
            raise RuntimeError("TSV header must contain column 'en'")
        if use_zh_col not in col:
            raise RuntimeError(f"TSV header must contain column '{use_zh_col}'")
        score_idx = col.get("score", None)

        for line in tqdm(f, desc="load TSV candidates", unit="lines", mininterval=5):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(col["en"], col[use_zh_col]):
                continue

            en = parts[col["en"]].strip()
            zh = parts[col[use_zh_col]].strip()
            if not en or not zh:
                continue

            en_l = en.lower()
            if en_l in EN_STOP:
                continue

            # priority：用 score 列（没有就默认 0）
            prio = 0.0
            if score_idx is not None and score_idx < len(parts):
                try:
                    prio = float(parts[score_idx])
                except Exception:
                    prio = 0.0

            en2zh[en_l] = zh
            en2prio[en_l] = prio

    return en2zh, en2prio

# =========================
# 命中逻辑（高性能）
# =========================

def split_terms(en2zh: Dict[str, str]) -> Tuple[set, List[str]]:
    """
    返回：
      simple_terms: set(en_lower)  # 仅字母数字下划线
      complex_terms: [en_lower]    # 含 - . 等，走子串匹配
    """
    simple_terms = set()
    complex_terms = []
    for en_l in en2zh.keys():
        if RE_SIMPLE.fullmatch(en_l):
            simple_terms.add(en_l)
        else:
            complex_terms.append(en_l)
    return simple_terms, complex_terms

def find_hits(src: str, simple_terms: set, complex_terms: List[str]) -> List[str]:
    """
    返回命中的 en_lower 列表（不去重不排序由外面处理）
    """
    hits: List[str] = []
    src_lower = src.lower()

    # 1) simple：token 化后做集合交集
    toks = set(RE_WORD_TOK.findall(src_lower))
    if toks:
        hits.extend(list(toks.intersection(simple_terms)))

    # 2) complex：子串匹配（数量通常不大）
    #    注意：这里是 O(len(complex_terms))，但 complex 一般远少于 simple
    for term in complex_terms:
        if term in src_lower:
            hits.append(term)

    return hits

# =========================
# 主流程
# =========================

def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    en2zh, en2prio = load_tsv_candidates(CAND_TSV, USE_ZH_COL)
    print("[load] candidates:", len(en2zh))

    simple_terms, complex_terms = split_terms(en2zh)
    print("[split] simple_terms:", len(simple_terms), "complex_terms:", len(complex_terms))

    per_term_kept = defaultdict(int)
    out_cnt = 0
    scan_cnt = 0
    bad_json = 0

    with open(TRAIN_JSONL, "r", encoding="utf-8") as fin, \
         open(OUT_JSONL, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="scan train & build sft seed (tsv-only)", unit="lines", mininterval=5):
            scan_cnt += 1
            if MAX_SCAN_LINES and scan_cnt > MAX_SCAN_LINES:
                break
            if MAX_OUT and out_cnt >= MAX_OUT:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            src = obj.get(SRC_FIELD, "")
            tgt = obj.get(TGT_FIELD, "")
            if not isinstance(src, str) or not src:
                continue
            if not isinstance(tgt, str) or not tgt:
                continue

            # 1) 找命中
            hits = find_hits(src, simple_terms, complex_terms)
            if not hits:
                continue

            # 2) 去掉已达上限的 term，并按 priority 选前 MAX_TERMS_PER_SAMPLE 个
            uniq = []
            seen = set()
            for h in hits:
                if h in seen:
                    continue
                seen.add(h)
                if MAX_SAMPLES_PER_TERM and per_term_kept[h] >= MAX_SAMPLES_PER_TERM:
                    continue
                if h in EN_STOP:
                    continue
                uniq.append(h)

            if not uniq:
                continue

            # 按 TSV score（priority）降序，优先选“更值得干预”的
            uniq.sort(key=lambda x: en2prio.get(x, 0.0), reverse=True)
            chosen = uniq[:MAX_TERMS_PER_SAMPLE]
            if not chosen:
                continue

            # 3) 构造 terms
            terms = {en: en2zh[en] for en in chosen}

            # 4) 写出
            for en in chosen:
                per_term_kept[en] += 1

            out_obj = {"source": src, "terms": terms, "target": tgt}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_cnt += 1

    print("done")
    print("scan_lines:", scan_cnt)
    print("bad_json_lines:", bad_json)
    print("out_samples:", out_cnt)
    print("out_jsonl:", OUT_JSONL)

    # 额外：输出 terms 覆盖情况 top20（便于 sanity check）
    top = sorted(per_term_kept.items(), key=lambda x: x[1], reverse=True)[:20]
    print("top_terms_by_kept:")
    for en, c in top:
        print(f"  {en}\t{c}\t=>\t{en2zh.get(en,'')}")


if __name__ == "__main__":
    main()
