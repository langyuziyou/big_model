# -*- coding: utf-8 -*-
"""纯 Python PDF 文本提取器（无需第三方 PDF 库）。

支持：FlateDecode 解压、Type0(CID) 字体 + ToUnicode CMap 映射、
内容流 Tj/TJ/' 文本算子、文本行按 Tm/Td 坐标换行。
适用于由 Word/WPS 等导出的文本型 PDF（本系统目标：简历 PDF）。
"""
import re
import zlib

# ---------------------------------------------------------------- 基础解析

def _parse_objects(data: bytes) -> dict:
    """解析 PDF 对象：返回 {对象号: 原始 body(不含 endobj)}"""
    objs = {}
    for m in re.finditer(rb'(\d+)\s+0\s+obj(.*?)endobj', data, re.DOTALL):
        objs[int(m.group(1))] = m.group(2)
    return objs


def _dict_and_stream(body: bytes):
    """把对象 body 拆成 (字典部分, 流原始字节或 None)。

    优先按 /Length 精确读取流长度（避免 rstrip 误删压缩流结尾字节）。
    """
    m = re.search(rb'stream(\r\n|\r|\n)', body)
    if not m:
        return body, None
    d = body[:m.start()]
    rest = body[m.end():]
    lm = re.search(rb'/Length\s+(\d+)', d)
    if lm:
        length = int(lm.group(1))
        s = rest[:length]
    else:
        e = rest.rfind(b'endstream')
        s = rest[:e].rstrip(b'\r\n')
    return d, s


def _decompress(s: bytes) -> bytes:
    if s is None:
        return b''
    try:
        return zlib.decompress(s)
    except Exception:
        return s


def _parse_cmap(stream: bytes) -> dict:
    """解析 ToUnicode CMap，返回 {字码(int): unicode码点(int)}。"""
    txt = stream.decode('latin1', 'replace')
    mapping = {}
    for bm in re.finditer(r'(\d+)\s+beginbfchar(.*?)endbfchar', txt, re.DOTALL):
        for c, u in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', bm.group(2)):
            mapping[int(c, 16)] = int(u, 16)
    for bm in re.finditer(r'(\d+)\s+beginbfrange(.*?)endbfrange', txt, re.DOTALL):
        for lo, hi, st in re.findall(
                r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', bm.group(2)):
            lo_, hi_, st_ = int(lo, 16), int(hi, 16), int(st, 16)
            for c in range(lo_, hi_ + 1):
                mapping[c] = st_ + (c - lo_)
    return mapping


def _resolve_font_cmap(font_obj: int, objects: dict) -> dict:
    """根据字体对象号，返回该字体的 ToUnicode CMap。"""
    d, _ = _dict_and_stream(objects.get(font_obj, b''))
    refs = []
    m = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', d)
    if m:
        refs.append(int(m.group(1)))
    for desc in re.findall(rb'/DescendantFonts\s*\[(.*?)\]', d, re.DOTALL):
        for ref in re.findall(rb'(\d+)\s+0\s+R', desc):
            dd, _ = _dict_and_stream(objects.get(int(ref), b''))
            m2 = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', dd)
            if m2:
                refs.append(int(m2.group(1)))
    for ref in refs:
        td, ts = _dict_and_stream(objects.get(ref, b''))
        stream = _decompress(ts)
        if stream:
            cmap = _parse_cmap(stream)
            if cmap:
                return cmap
    return {}


# ---------------------------------------------------------------- 内容流解析

def _tokenize(s: str):
    pos = 0
    n = len(s)
    while pos < n:
        c = s[pos]
        if c in ' \t\r\n\f':
            pos += 1
            continue
        if c == '/':
            m = re.match(r'/[A-Za-z0-9.+\-]*', s[pos:])
            yield m.group(0)
            pos += len(m.group(0))
            continue
        if c in '[]':
            yield c
            pos += 1
            continue
        if c == '<':
            if s[pos:pos + 2] == '<<':
                yield '<<'
                pos += 2
                continue
            m = re.match(r'<[0-9A-Fa-f\s]*>', s[pos:])
            yield m.group(0)
            pos += len(m.group(0))
            continue
        if c == '>':
            yield '>>'
            pos += 2
            continue
        if c == '(':
            m = re.match(r'\((?:\\.|[^()\\])*\)', s[pos:])
            yield m.group(0)
            pos += len(m.group(0))
            continue
        if c.isdigit() or c in '+-.':
            m = re.match(r'[+-]?\d*\.?\d+', s[pos:])
            yield m.group(0)
            pos += len(m.group(0))
            continue
        m = re.match(r"[A-Za-z'\"*]+", s[pos:])
        if m:
            yield m.group(0)
            pos += m.end()
            continue
        pos += 1


def _iter_array_strings(arr_raw: str):
    for m in re.finditer(r'<([0-9A-Fa-f]+)>|\((?:\\.|[^()\\])*\)', arr_raw):
        yield m.group(0)


def _decode_literal(tok: str) -> str:
    inner = tok[1:-1]
    out = []
    i = 0
    while i < len(inner):
        c = inner[i]
        if c == '\\' and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt in '()\\':
                out.append(nxt)
                i += 2
            elif nxt == 'n':
                out.append('\n'); i += 2
            elif nxt == 'r':
                out.append('\r'); i += 2
            elif nxt == 't':
                out.append('\t'); i += 2
            elif nxt == 'b':
                out.append('\b'); i += 2
            elif nxt == 'f':
                out.append('\f'); i += 2
            elif nxt.isdigit():
                m = re.match(r'[0-7]{1,3}', inner[i + 1:])
                out.append(chr(int(m.group(0), 8)))
                i += 1 + len(m.group(0))
            else:
                out.append(nxt)
                i += 2
        else:
            out.append(c)
            i += 1
    s = ''.join(out)
    if s.startswith('\ufeff'):
        s = s[1:]
    if '\x00' in s:
        try:
            s = s.encode('latin1').decode('utf-16-be')
        except Exception:
            pass
    return s


def _decode_hex(tok: str, cur_font, font_cmaps: dict) -> str:
    hexs = tok[1:-1].strip().replace(' ', '')
    if len(hexs) % 4 != 0:
        hexs = hexs.ljust((len(hexs) // 4 + 1) * 4, '0')
    cmap = font_cmaps.get(cur_font, {})
    out = []
    for i in range(0, len(hexs), 4):
        code = int(hexs[i:i + 4], 16)
        if code in cmap:
            out.append(chr(cmap[code]))
        elif code < 128:
            out.append(chr(code))
        else:
            out.append('')  # 未知字形，丢弃
    return ''.join(out)


def _extract_content(content: bytes, font_cmaps: dict) -> str:
    s = content.decode('latin1')
    tokens = list(_tokenize(s))
    out = []
    stack = []
    cur_font = None
    x = y = 0.0
    last_y = None
    newline_pending = False

    def show(tok: str):
        nonlocal last_y, newline_pending
        decoded = _decode_hex(tok, cur_font, font_cmaps) if tok.startswith('<') \
            else _decode_literal(tok)
        if not decoded:
            return
        if newline_pending:
            out.append('\n')
            newline_pending = False
            last_y = None
        elif last_y is not None and abs(y - last_y) > 2.0:
            out.append('\n')
        out.append(decoded)
        last_y = y

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == '<<':
            depth = 0
            while i < n:
                if tokens[i] == '<<':
                    depth += 1
                elif tokens[i] == '>>':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            i += 1
            continue
        if tok == '[':
            buf = []
            i += 1
            while i < n and tokens[i] != ']':
                buf.append(tokens[i])
                i += 1
            stack.append(' '.join(buf))
            i += 1  # 跳过 ']'
            continue
        if tok == ']':
            i += 1
            continue
        if tok == 'Tf':
            if len(stack) >= 2:
                size = stack.pop()
                name = stack.pop()
                cur_font = name[1:] if name.startswith('/') else name
            stack.clear()
            i += 1
            continue
        if tok == 'Tm':
            if len(stack) >= 6:
                f = float(stack.pop())
                e = float(stack.pop())
                x, y = e, f
            stack.clear()
            i += 1
            continue
        if tok in ('Td', 'TD'):
            if len(stack) >= 2:
                ty = float(stack.pop())
                tx = float(stack.pop())
                x += tx
                y += ty
            stack.clear()
            i += 1
            continue
        if tok == 'T*':
            newline_pending = True
            stack.clear()
            i += 1
            continue
        if tok == 'Tj':
            if stack:
                show(stack.pop())
            stack.clear()
            i += 1
            continue
        if tok == 'TJ':
            if stack:
                arr = stack.pop()
                for st in _iter_array_strings(arr):
                    show(st)
            stack.clear()
            i += 1
            continue
        if tok == "'":
            if stack:
                show(stack.pop())
            stack.clear()
            i += 1
            continue
        if tok == '"':
            if len(stack) >= 3:
                show(stack[-1])
            stack.clear()
            i += 1
            continue
        # 其它算子/操作数：数字、名称、字符串压栈；算子消费后清空
        if re.match(r"^[A-Za-z'\"*]+$", tok):
            stack.clear()  # 未知算子，清空操作数
        else:
            stack.append(tok)
        i += 1
    return ''.join(out)


# ---------------------------------------------------------------- 顶层入口

def _page_order(objects: dict) -> list:
    """按页面树 Kids 深度优先得到页面对象号顺序。"""
    root = None
    for num, body in objects.items():
        if re.search(rb'/Type\s*/Pages', body) and b'/Parent' not in body:
            root = num
            break
    if root is None:
        # 兜底：按对象号顺序里的 /Type/Page
        return [n for n, b in objects.items() if re.search(rb'/Type\s*/Page\b', b)]
    order = []

    def visit(num):
        body = objects.get(num, b'')
        if re.search(rb'/Type\s*/Page\b', body):
            order.append(num)
            return
        km = re.search(rb'/Kids\s*\[(.*?)\]', body, re.DOTALL)
        if km:
            for ref in re.findall(rb'(\d+)\s+0\s+R', km.group(1)):
                visit(int(ref))

    visit(root)
    return order


def extract_text(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as f:
        data = f.read()
    objects = _parse_objects(data)

    font_cmaps = {}
    # 从所有页面/资源的 /Font 字典建立 字体资源名 -> CMap
    for num, body in objects.items():
        fm = re.search(rb'/Font\s*<<(.*?)>>', body, re.DOTALL)
        if not fm:
            continue
        for name, ref in re.findall(rb'/([A-Za-z0-9]+)\s+(\d+)\s+0\s+R', fm.group(1)):
            cmap = _resolve_font_cmap(int(ref), objects)
            if cmap:
                font_cmaps[name.decode('ascii')] = cmap

    pages = []
    for page_obj in _page_order(objects):
        body = objects.get(page_obj, b'')
        cm = re.search(rb'/Contents\s+(\d+)\s+0\s+R', body)
        if cm:
            refs = [int(cm.group(1))]
        else:
            am = re.search(rb'/Contents\s*\[(.*?)\]', body, re.DOTALL)
            refs = [int(x) for x in re.findall(rb'(\d+)\s+0\s+R', am.group(1))] if am else []
        for ref in refs:
            _, stream = _dict_and_stream(objects.get(ref, b''))
            content = _decompress(stream)
            if content:
                pages.append(_extract_content(content, font_cmaps))

    text = '\n'.join(pages)
    # 清理：压缩多余空行
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
