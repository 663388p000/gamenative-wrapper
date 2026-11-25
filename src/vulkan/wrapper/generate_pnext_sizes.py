import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if len(sys.argv) != 3:
    raise RuntimeError("Usage: generate_pnext_sizes.py vk.xml output.inc")

root = ET.parse(sys.argv[1]).getroot()
header = Path("include/vulkan/vulkan_core.h").read_text(errors="ignore")

# Resolve Vulkan enum aliases to canonical names.
aliases = {}

for enum in root.findall(".//enum"):
    name = enum.get("name")
    alias = enum.get("alias")
    if name and alias:
        aliases[name] = alias

def canonical(name):
    seen = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name

entries = {}
conflicts = []

for struct in root.findall("./types/type[@category='struct']"):
    name = struct.get("name") or struct.findtext("name")

    if not name or struct.get("alias"):
        continue

    if "VkDeviceCreateInfo" not in struct.get("structextends", "").split(","):
        continue

    members = {
        m.findtext("name"): m
        for m in struct.findall("member")
    }

    if "sType" not in members or "pNext" not in members:
        continue

    stype = members["sType"].get("values", "").strip()

    if not re.fullmatch(r"VK_STRUCTURE_TYPE_[A-Z0-9_]+", stype):
        continue

    # Require the actual type declaration in vulkan_core.h.
    if not re.search(
        r"}\s*" + re.escape(name) + r"\s*;",
        header
    ):
        continue

    # Require the enum identifier in the same header.
    if not re.search(r"\b" + re.escape(stype) + r"\b", header):
        continue

    key = canonical(stype)
    entry = (stype, name, struct.get("protect"))

    if key in entries:
        old = entries[key]
        if old[1] != name:
            # Keep only a type actually declared in vulkan_core.h.
            continue
    else:
        entries[key] = entry

if not entries:
    raise RuntimeError("No compatible pNext structures found")

lines = [
    "/* Generated from matching Vulkan registry and core header. */",
    "/* Do not edit manually. */",
    ""
]

for stype, name, protect in sorted(entries.values()):
    if protect:
        lines.append(f"#ifdef {protect}")

    lines.append(f"   case {stype}: return sizeof({name});")

    if protect:
        lines.append("#endif")

    lines.append("")

out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines))

print("Generated structures:", len(entries))
print("Output:", out)
