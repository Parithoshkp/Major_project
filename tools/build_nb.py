"""Convert a '# %%' percent-format script into a .ipynb (markdown cells: '# %% [markdown]')."""
import re
import sys

import nbformat as nbf

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
cells = []
for block in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in header:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(nbf.v4.new_markdown_cell(md))
    else:
        cells.append(nbf.v4.new_code_cell(body))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "tfet-ml", "display_name": "Python (TFET-ML)", "language": "python"}
nbf.write(nb, dst)
print(f"wrote {dst} ({len(cells)} cells)")
