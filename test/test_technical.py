"""Technical corpus ingestion must not train on stale or untranslated entries."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "prepare_technical", Path(__file__).resolve().parents[1] / "neural/prepare-technical.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Translations(unittest.TestCase):
    def test_only_completed_translations_become_prose(self):
        source = '''msgid ""
msgstr "Language: zh_CN\\n"

msgid "compile"
msgstr "重新"
"编译代码"

#, fuzzy
msgid "thread"
msgstr "不应使用的旧翻译"

#~ msgid "removed"
#~ msgstr "已经删除的翻译"

msgid "untranslated"
msgstr ""

msgid "quote"
msgstr "保留 \\"引号\\" 和换行\\n下一行"
'''
        self.assertEqual(module.translated_prose(source),
                         '重新编译代码\n保留 "引号" 和换行\n下一行')


if __name__ == "__main__":
    unittest.main()
