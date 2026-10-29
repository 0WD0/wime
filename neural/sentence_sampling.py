"""Conservative, inspectable boundaries and legacy span controls for sentence training."""
import random
import re

LENGTHS = ((.50, 1, 1), (.25, 2, 3), (.15, 4, 8), (.10, 9, 24))
PAIRS = (("如果", "就"), ("只要", "就"), ("只有", "才"), ("因为", "所以"),
         ("虽然", "但是"), ("既然", "就"), ("不仅", "而且"), ("先", "再"))
PREDICATES = re.compile("需要|可以|应该|必须|已经|正在|能够|是否|怎么|如何|是不是|不能|不会|有什么")
REQUEST = re.compile("^(?:请|求|想|希望|帮|告诉|有没有|能不能|如何|怎么|为什么)")


def source_units(text, minimum=8, maximum=60):
    """Exact source sentences, or clauses ending at existing punctuation.

    Never remove an internal symbol, concatenate across punctuation, or cut a
    long Han run just to fit the model window.
    """
    def unit(raw, offset, kind):
        target = raw.rstrip('。！？!?\n，；').strip()
        start = offset + len(raw) - len(raw.lstrip())
        terminal = raw[-1] if raw[-1] in '。！？!?，；' else ''
        if (minimum <= len(target) <= maximum and re.fullmatch(r'[\u3400-\u4dbf\u4e00-\u9fff]+',target)
                and len(set(target)) >= 4 and not re.search(r'(的话|但是|而且|[把被与或但不就才该])$',target)
                and text[start:start+len(target)+len(terminal)] == target+terminal):
            return dict(offset=start,text=target,terminal=terminal,unit_type=kind)

    for sentence in re.finditer(r'[^。！？!?\n]+[。！？!?\n]?',text):
        whole = unit(sentence.group(),sentence.start(),'sentence')
        if whole:
            yield whole
        elif re.search('[，；]',sentence.group()):
            for clause in re.finditer(r'[^，；]+[，；]?',sentence.group()):
                item = unit(clause.group(),sentence.start()+clause.start(),'clause')
                if item:
                    yield item


def retain_group(groups, row, limit):
    """Hash-reservoir per source group and length band, independent of boundaries."""
    key = (row['group'],len(row['gold'][0])>24)
    bucket = groups.setdefault(key,[])
    if len(bucket)<limit or row['id']<bucket[-1]['id']:
        bucket.append(row)
        bucket.sort(key=lambda r:r['id'])
        del bucket[limit:]


def boundary(text, segments):
    """Return at most one rule-supported boundary; no arbitrary character cut."""
    ends, tags, position = {}, [], 0
    for word, pos in segments:
        position += len(word)
        ends[position] = pos
        tags.append((position,pos))
    if position != len(text):
        raise ValueError("segmentation changed source text")
    for left, right in PAIRS:
        if text.startswith(left):
            end = text.find(right, len(left) + 2)
            prefix = text[:end]
            if (4 <= end <= min(14,len(text)-3) and end in ends
                    and not ends[end].startswith("r")
                    and not re.search("可能|应该|那么|真的|是不是|怎么|如何|恐怕|那$|要$|才$", prefix)):
                return [dict(end=end, reason=f"成对连接结构：{left}…｜{right}…", rule="paired-clause")]
    for match in re.finditer("之后|以后|的时候|之前", text):
        end = match.end()
        if (4 <= end <= min(14, len(text)-3) and end in ends and not text[end:].startswith("的")
                and any(pos.startswith(("v","t")) for stop,pos in tags if stop<=match.start())
                and not re.search("希望|知道|想|说|大概|感觉|觉得|讨厌|特别烦|问",text[:end])):
            return [dict(end=end, reason="完整时间条件之后", rule="time-clause")]
    if not REQUEST.match(text):
        for match in PREDICATES.finditer(text):
            end = match.start()
            prefix = text[:end]
            if (3 <= end <= len(text)-4 and ends.get(end, "").startswith(("n", "s"))
                    and all(pos.startswith(("n","r","a","m","q","f","s","t","u"))
                            for stop,pos in tags if stop<=end)
                    and not re.search("不知道|希望|觉得|感觉|认为|我就|如果|什么|怎么|求问|请问|带了|[该这那]$", prefix)):
                return [dict(end=end, reason="名词性话题之后，后接明确谓语或疑问", rule="topic-predicate")]
    return []


def legacy_span(text, spans, seed):
    """Same length lottery and word alignment as the old corpus, one draw per source."""
    rng = random.Random(seed)
    starts = [start for start, end, word in spans]
    ends = [end for start, end, word in spans]
    start = rng.choice(starts)
    draw, total = rng.random(), 0
    for share, low, high in LENGTHS:
        total += share
        if draw < total:
            length = rng.randint(low, high)
            break
    end = next((end for end in ends if end >= start+length), len(text))
    return [start, end]
