"""Source units, commit-boundary candidates and legacy span controls for sentence training."""
import random
import re

LENGTHS = ((.50, 1, 1), (.25, 2, 3), (.15, 4, 8), (.10, 9, 24))


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


def technical_line_fragment(text, start):
    """A documentation soft wrap is not a new composition boundary."""
    before=text[:start]
    return bool(before and before[-1].isspace() and before.rstrip()
                and before.rstrip()[-1] not in '。！？!?，；：')


NOMINAL = ("n", "vn", "an", "s", "t", "f", "m", "q", "j", "l", "k", "eng")
PREDICATE_WORDS = {"是", "有", "能", "会", "要", "可以", "应该", "需要", "得", "该", "还是", "就是", "不是", "没有",
                   "怎么", "怎么样", "如何", "为什么", "多少", "哪", "哪里", "哪个", "吗", "算", "叫", "像", "让"}
CONDITION_ENDS = ("的话", "之后", "以后", "以前", "之前", "的时候", "时候", "后", "时", "中", "里", "期间", "以来", "以外", "外")
CONDITION_STARTS = ("如果", "要是", "假如", "只要", "即使", "虽然", "因为", "既然", "一旦", "除非", "就算", "哪怕")
MAIN_MARKERS = ("就", "才", "也", "还", "那", "所以", "但", "可是", "却", "都")
REQUEST = re.compile("^(?:请|求|帮|给我|写|告诉|有没有|能不能|能否|可不可以|如何|怎么|为什么|我想|我要|你能|你可以|麻烦)")
PRONOUNS = set("我你他她它您咱") | {"我们", "你们", "他们", "她们", "咱们", "大家", "自己", "这", "那", "这个", "那个"}
FOLLOW_CONJUNCTIONS = {"还是", "然后", "但", "但是", "可是", "所以", "结果", "于是", "而且", "不过", "却", "就", "也", "还"}
INTERROGATIVE = re.compile("^(?:怎|咋|啥|什么|为什么|为啥|哪|谁|几|多少|多久)")
FRAMES_START = re.compile("^(?:被|把|是|不是|还是|就是|是不是|有没有)")
FRAMES_END = re.compile("(?:是不是|是否|有没有|能不能|会不会|要不要|可不可以|吗|呢|一点|一些|一下|一会儿"
                        "|知道|说明|觉得|认为|发现|看到|听说|希望|以为|感觉|担心|怀疑|表示|是|不是|就是"
                        "|怎么|怎样|为什么|这么|那么|多么|有所|无法|可以|能够|应该|需要|必须|连|都|也|就|还|那|这)$")
MODIFIER_SUFFIX = re.compile("^[^的，]{1,12}的[\u4e00-\u9fff]")
COGNITIVE = re.compile("认定|觉得|认为|知道|发现|告诉|听说|希望|以为|感觉|确定|相信|怀疑|担心|看到|听到|想到|导致|使得|建议")
COVERBS = re.compile("(?:将|把|被|对|向|给|为|因为|由于|让|使|用|以|从|跟|和|与)")
TIME_FRAMES = re.compile("^(?:等到|直到|当|每当|自从|自打)")
COORDINATION = re.compile("^(?:与|和|及|以及|或|或者|跟|并)")
ADVERBISH = {"有点", "有些", "好", "很", "太", "真", "挺", "超", "特别", "非常", "总是", "一直", "经常", "已经", "也是", "都是"}
VERBAL = ("v", "l", "i")
FUNCTION = ("d", "p", "c", "u", "r", "y", "e", "f")
ATTACHED = re.compile("^(?:的|地|得|了|着|过|们|之|么|吗|呢|吧|啊|呀|嘛|性|化|者|上|下|里|中)")


def opens_predicate(word, flag):
    return (word in PREDICATE_WORDS or bool(INTERROGATIVE.match(word))
            or flag.startswith(("d", "v", "a", "p", "t")) and flag != "vn")


def boundary_candidates(text, segments, lexicon_ends=None):
    """Every grammatically finished commit point as (rule, end); commit_pause picks one.

    A finished topic, condition, time phrase or statement must precede, and the
    rest must start a predicate or follow-up. Rejections mirror dev-v2 review:
    unfinished objects, comparisons, quantity frames, open prepositions,
    request stubs, and cuts that strand 的/了/们. ``lexicon_ends`` are word ends
    from the wime dictionary segmentation; they veto jieba splits next to
    single-character content tokens (通｜选课).
    """
    ends, words, position = [], [], 0
    for word, flag in segments:
        position += len(word)
        ends.append(position)
        words.append((word, flag))
    if position != len(text):
        return []
    found = []
    for k in range(len(words) - 1):
        end = ends[k]
        prefix, suffix = text[:end], text[end:]
        if not (4 <= end <= 16 and len(suffix) >= 3) or ATTACHED.match(suffix) or COORDINATION.match(suffix):
            continue
        # Single-character tokens are where jieba splits words (通｜选课, 中｜长发);
        # there the dictionary segmentation must agree.
        if lexicon_ends is not None and end not in lexicon_ends and any(
                len(w) == 1 and not f.startswith(FUNCTION) and w not in PREDICATE_WORDS and w not in MAIN_MARKERS
                for w, f in (words[k], words[k + 1])):
            continue
        if TIME_FRAMES.match(prefix) and not prefix.endswith(CONDITION_ENDS):
            continue
        # The object clause of a cognitive verb is still open: 认定一个女生｜不可能接受他.
        if COGNITIVE.search(prefix) and not prefix.endswith(CONDITION_ENDS) and not re.search(
                "(?:" + COGNITIVE.pattern + ")[^，]*(?:" + "|".join(sorted(PREDICATE_WORDS - {"吗"})) + ")", prefix):
            continue
        if FRAMES_START.match(prefix) or FRAMES_END.search(prefix) or REQUEST.fullmatch(prefix) or (
                REQUEST.match(prefix) and len(prefix) <= 5):
            continue
        last_word, last_flag = words[k]
        next_word, next_flag = words[k + 1]
        head = words[:k + 1]
        if k and words[k - 1][0] == "的" and last_flag.startswith("v"):
            last_flag = "vn"  # nominalized after 的: 考试的复习
        if next_word in ADVERBISH:
            next_flag = "d"
        # A prepositional phrase inside a clause still needs its verb: 没有从事物中｜归纳.
        if prefix.endswith(("中", "里", "上")) and re.search("[从在于对][^，]*$", prefix[1:]):
            continue
        # An open prepositional frame needs its verb before the boundary: 对国有企业｜以….
        # A two-character "preposition" ending the prefix is a verb: 没通过, 不对.
        last_preposition = max((i for i, (w, f) in enumerate(head) if (f.startswith("p") or COVERBS.fullmatch(w))
                                and not (i == k and len(w) >= 2)), default=None)
        if last_preposition is not None and not any(f.startswith(VERBAL) and f != "vn"
                                                     for _, f in head[last_preposition + 1:]):
            if not (last_preposition == 0 and prefix.endswith(CONDITION_ENDS)):
                continue
        if last_flag.startswith(("c", "d")) or prefix in PRONOUNS or all(w in PRONOUNS or f.startswith(("c", "d", "p", "u", "y")) for w, f in head):
            continue
        # Review rejections: unfinished comparison, quantity frame, dangling preposition.
        if re.search("(比|跟|和|与|像|对|把|被|给|向|从|在|为|一)$", prefix) or re.search("比[^，]{0,3}$", prefix):
            continue
        weak_modifier = MODIFIER_SUFFIX.match(suffix)
        strong = (next_word in PREDICATE_WORDS or next_word in MAIN_MARKERS or next_word in FOLLOW_CONJUNCTIONS
                  or bool(INTERROGATIVE.match(next_word)) or next_flag.startswith("d"))
        if prefix.startswith(CONDITION_STARTS):
            core = re.sub("的话$", "", prefix[next(len(c) for c in CONDITION_STARTS if prefix.startswith(c)):])
            if (strong or next_word in PRONOUNS) and len(core) >= 2 and core not in ("是", "这样", "那样", "这样的", "那样的"):
                found.append(("condition-clause", end))
            continue
        if all(f.startswith("t") for _, f in head) and (next_word in PRONOUNS or next_flag.startswith(("n", "r"))):
            found.append(("time-condition", end))
            continue
        if prefix.endswith(CONDITION_ENDS) and any(f.startswith(("v", "n", "l", "i")) for _, f in head[:-1]):
            if weak_modifier and not strong:
                continue  # 之前提及的换行符: the marker modifies a noun
            if opens_predicate(next_word, next_flag) or next_word in PRONOUNS or next_flag.startswith("n"):
                found.append(("time-condition", end))
            continue
        if last_flag.startswith(NOMINAL) and opens_predicate(next_word, next_flag):
            # A nominal topic: no finite verb after the last 的, so "有梦想的人" stays one topic.
            # A relative clause before 的 is part of the topic ("有梦想的人"); a bare
            # verb phrase without 的 is a statement, handled below.
            if "的" in prefix or not any(f.startswith("v") and f != "vn" for _, f in head):
                if strong or not weak_modifier:
                    found.append(("topic-predicate", end))
                continue
        new_subject = (next_word in PRONOUNS - {"这", "那", "这个", "那个", "自己"} and not last_flag.startswith("v")
                       and k + 2 < len(words) and words[k + 2][0] != "的"
                       and words[k + 2][1].startswith(("v", "d", "a", "p")))
        if weak_modifier and not strong:
            continue
        if any(f.startswith(VERBAL) for _, f in head) and len(prefix) >= 5 and not weak_modifier and (
                next_word in FOLLOW_CONJUNCTIONS or new_subject
                or next_flag.startswith("d") and (last_word in ("了", "过") or last_flag.startswith(("a", "y", "n", "m")))):
            found.append(("statement-follow", end))
    return found


REASONS = {
    "condition-clause": "完整条件或让步分句之后，后接主句",
    "time-condition": "时间、条件或范围短语结束，后文仍有实质内容",
    "topic-predicate": "名词性话题结束，后接谓语、判断或疑问",
    "statement-follow": "前一陈述完成，后接副词或主语引出的后续陈述"}


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
