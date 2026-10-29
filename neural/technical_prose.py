"""Pinned technical prose rendering, shared by legacy and contextual datasets."""
import ast
import re
import tarfile

def translated_prose(text):
    """Read completed gettext translations, excluding fuzzy/obsolete entries."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        if re.search(r"^#,.*\bfuzzy\b|^#~", block, re.M):
            continue
        fields, field = {}, None
        for line in block.splitlines():
            match = re.match(r'(msgid|msgstr) (".*")$', line)
            if match:
                field = match[1]
                fields[field] = ast.literal_eval(match[2])
            elif line.startswith('"') and field:
                fields[field] += ast.literal_eval(line)
            elif not line.startswith("#"):
                field = None
        if fields.get("msgid") and fields.get("msgstr"):
            paragraphs.append(fields["msgstr"])
    return "\n".join(paragraphs)


def render(text, python=False):
    if python:
        text = translated_prose(text)
        text = re.sub(r":[a-z]+:`([^`<>]+)(?: <[^`]+>)?`", r"\1", text)
        text = re.sub(r"\n(?:[ \t]{2,}[^\n]*(?:\n|$))+", "\n", text)
    text = re.sub(r"\A---\n.*?\n---\n|<!--.*?-->|```.*?```|{{.*?}}", " ", text, flags=re.S)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(r"\n\s*\|.*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Soft line wraps in Chinese prose should not split its meaning.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
    return text


def documents(archive, source, revision):
    root=("python-docs-zh-cn-" if source=="python" else "website-")+revision
    with tarfile.open(archive,"r|gz") as stream:
        for member in stream:
            if member.name.split("/",1)[0]!=root:
                raise ValueError("archive does not match pinned revision")
            if not member.isfile():continue
            if source=="python":
                if not member.name.endswith(".po") or not any("/"+part+"/" in member.name for part in
                        ("tutorial","library","reference","howto","c-api","extending","using","installing","builtins","faq")):
                    continue
            elif "/content/zh-cn/docs/" not in member.name or not member.name.endswith(".md") or "/reference/" in member.name:
                continue
            raw=stream.extractfile(member).read().decode()
            yield member.name,raw,render(raw,source=="python")
