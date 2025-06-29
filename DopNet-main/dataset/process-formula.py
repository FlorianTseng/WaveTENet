import pandas as pd
import re
from collections import defaultdict

def expand_formula(formula: str) -> str:
    """
    Expand formulas like 'BiSb(Se0.92Br0.08)3' into 'Bi1Sb1Se2.76Br0.24'
    """
    elements = defaultdict(float)

    # 拆主成分（括号前部分）
    base_part = re.findall(r'[A-Z][a-z]?[0-9.]*', formula.split('(')[0])
    for token in base_part:
        m = re.match(r'([A-Z][a-z]?)([0-9.]*)', token)
        if m:
            el, amt = m.group(1), float(m.group(2)) if m.group(2) else 1.0
            elements[el] += amt

    # 拆括号里的掺杂部分（如 Se0.92Br0.08）3
    mix_blocks = re.findall(r'\((.*?)\)([0-9.]*)', formula)
    for block, multiplier in mix_blocks:
        multiplier = float(multiplier) if multiplier else 1.0
        parts = re.findall(r'([A-Z][a-z]?)([0-9.]*)', block)
        for el, amt in parts:
            amt = float(amt) if amt else 1.0
            elements[el] += amt * multiplier

    # 按元素字母顺序生成字符串
    expanded = ''.join([f'{el}{elements[el]:.4g}' for el in sorted(elements.keys())])
    return expanded

# === 读取 Excel 文件 ===
df = pd.read_excel("new.xlsx")

# === 扩展 Formula 列 ===
df["Formula_expanded"] = df["Formula"].apply(expand_formula)

# === 保存新的 Excel 文件 ===
df.to_excel("expanded_formula.xlsx", index=False)
