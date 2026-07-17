#!/usr/bin/env python3
"""One-off helper: derive Makefile VERILOG_SOURCES/EXTRA_ARGS blocks from
FuseSoC's generated .eda.yml manifest, and diff the file basenames against
the old vendored tree's Makefile source list."""
import yaml
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "_fusesoc_build_manifest.eda.yml")

with open(MANIFEST) as f:
    data = yaml.safe_load(f)

files = data["files"]

verilog_srcs = []   # (path, is_include)
vlt_files = []
include_dirs = []   # directories containing .svh include-only files, deduped

for entry in files:
    ftype = entry.get("file_type", "")
    name = entry["name"]  # relative to build dir, e.g. src/xxx/rtl/foo.sv
    if name.startswith("src/"):
        name = name[len("src/"):]  # we copied src/*'s CONTENTS into src_upstream/, flattening this prefix
    is_incl = entry.get("is_include_file", False)
    if ftype == "vlt":
        vlt_files.append(name)
    elif ftype == "systemVerilogSource" or ftype == "verilogSource":
        if is_incl:
            d = os.path.dirname(name)
            if d not in include_dirs:
                include_dirs.append(d)
        else:
            verilog_srcs.append(name)
    elif ftype == "user":
        pass  # python tooling files, not needed for the Verilator build
    else:
        print(f"# NOTE unhandled file_type={ftype}: {name}", file=sys.stderr)

print(f"# {len(vlt_files)} .vlt waivers, {len(verilog_srcs)} sv sources, {len(include_dirs)} include-only dirs", file=sys.stderr)

# --- Emit Makefile.upstream fragment ---
out = []
out.append("VERILOG_SOURCES += \\")
lines = []
for p in vlt_files:
    lines.append(f"\t$(PWD)/src_upstream/{p}")
for p in verilog_srcs:
    lines.append(f"\t$(PWD)/src_upstream/{p}")
lines.append("\t$(PWD)/cocotb_ibex_max_upstream.sv")
out.append(" \\\n".join(lines))
out.append("")
out.append("EXTRA_ARGS += \\")
extra_lines = []
for d in include_dirs:
    extra_lines.append(f"\t+incdir+$(PWD)/src_upstream/{d} \\\n\t-CFLAGS -I$(PWD)/src_upstream/{d}")
out.append(" \\\n".join(extra_lines))

with open(os.path.join(HERE, "_generated_sources.mk"), "w") as f:
    f.write("\n".join(out) + "\n")

# --- Diff basenames vs old Makefile ---
old_makefile = os.path.join(HERE, "..", "Makefile")
with open(old_makefile) as f:
    old_content = f.read()
old_basenames = set(re.findall(r"([A-Za-z0-9_.]+\.(?:sv|svh|vlt))\b", old_content))

new_basenames = set(os.path.basename(p) for p in vlt_files + verilog_srcs)

only_old = sorted(old_basenames - new_basenames)
only_new = sorted(new_basenames - old_basenames)

with open(os.path.join(HERE, "_diff_report.txt"), "w") as f:
    f.write("=== Files in OLD vendor Makefile but NOT in fresh upstream manifest ===\n")
    f.write("\n".join(only_old) + "\n\n")
    f.write("=== Files in fresh upstream manifest but NOT in OLD vendor Makefile ===\n")
    f.write("\n".join(only_new) + "\n")

print("Wrote _generated_sources.mk and _diff_report.txt")
print(f"only_old={len(only_old)} only_new={len(only_new)}")
