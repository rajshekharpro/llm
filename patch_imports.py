# patch_imports.py
import os
root = os.path.abspath(os.path.dirname(__file__))
old = "word_tokenizer.tokenizer"
new = "word_tokenizer.tokenizer"

for subdir, dirs, files in os.walk(root):
    for fname in files:
        if fname.endswith((".py", ".ipynb")):
            path = os.path.join(subdir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
            except:
                continue
            if old in txt:
                newtxt = txt.replace(old, new)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(newtxt)
                print("Patched:", path)

