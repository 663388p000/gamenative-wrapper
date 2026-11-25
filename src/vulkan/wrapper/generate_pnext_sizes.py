import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: generate_pnext_sizes.py vk.xml output.inc"
    )

xml_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

header_dir = Path("include/vulkan")
if not xml_path.is_file() or not header_dir.is_dir():
    raise RuntimeError("Registry or Vulkan headers not found")

headers = "\n".join(
    p.read_text(errors="ignore")
    for p in header_dir.glob("*.h")
)

root = ET.parse(xml_path).getroot()
entries = {}
conflicts = []

for struct in root.findall("./types/type[@category='struct']"):
    name = struct.get("name") or struct.findtext("name")

    if not name or struct.get("alias"):
        continue

    extends = struct.get("structextends", "").split(",")
    if "VkDeviceCreateInfo" not in extends:
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

    # Do not generate references absent from this checkout's headers.
    if not re.search(r"\b" + re.escape(name) + r"\b", headers):
        continue

    if not re.search(r"\b" + re.escape(stype) + r"\b", headers):
        continue

    entry = (name, struct.get("protect"))

    if stype in entries and entries[stype] != entry:
        conflicts.append((stype, entries[stype], entry))
    else:
        entries[stype] = entry

if conflicts:
    raise RuntimeError(f"Conflicting sTypes: {conflicts}")

if not entries:
    raise RuntimeError("No compatible VkDeviceCreateInfo structures found")

lines = [
    "/* Generated from this checkout's Vulkan registry. */",
    "/* Do not edit manually. */",
    ""
]

for stype, (name, protect) in sorted(entries.items()):
    if protect:
        lines.append(f"#ifdef {protect}")

    lines.append(f"   case {stype}: return sizeof({name});")

    if protect:
        lines.append("#endif")

    lines.append("")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines))

print("Generated:", out_path)
print("Structures:", len(entries))
