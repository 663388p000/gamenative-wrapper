import xml.etree.ElementTree as ET
from pathlib import Path
import re
import sys

xml_path = Path.home() / "storage/downloads/vk.xml"
out_path = Path("wrapper_pnext_sizes.inc")

if not xml_path.is_file():
    print("ERROR: vk.xml not found:", xml_path)
else:
    root = ET.parse(xml_path).getroot()
    entries = {}
    conflicts = []
    skipped = 0

    for struct in root.findall("./types/type[@category='struct']"):
        name = struct.get("name")

        if not name:
            name = struct.findtext("name")

        if not name or not name.startswith("Vk"):
            skipped += 1
            continue

        if struct.get("alias"):
            skipped += 1
            continue

        extends = struct.get("structextends", "").split(",")
        if "VkDeviceCreateInfo" not in extends:
            skipped += 1
            continue

        members = {
            member.findtext("name"): member
            for member in struct.findall("member")
        }

        if "sType" not in members or "pNext" not in members:
            skipped += 1
            continue

        stype = members["sType"].get("values", "").strip()

        if not re.fullmatch(r"VK_STRUCTURE_TYPE_[A-Z0-9_]+", stype):
            skipped += 1
            continue

        entry = (name, struct.get("protect"))

        if stype in entries and entries[stype] != entry:
            conflicts.append((stype, entries[stype], entry))
        else:
            entries[stype] = entry

    if conflicts:
        print("ERROR: sType conflicts:")
        for item in conflicts:
            print(item)

    elif not entries:
        print("ERROR: No VkDeviceCreateInfo extensions found.")

    else:
        lines = [
            "/* Generated from vk.xml. */",
            "/* Only VkDeviceCreateInfo.pNext structures. */",
            ""
        ]

        for stype, (name, protect) in sorted(entries.items()):
            if protect:
                lines.append(f"#ifdef {protect}")

            lines.append(f"   case {stype}: return sizeof({name});")

            if protect:
                lines.append("#endif")

            lines.append("")

        out_path.write_text("\n".join(lines))

        print("GENERATION SUCCESS")
        print("Structures:", len(entries))
        print("Skipped:", skipped)
        print("Protected:", sum(bool(p) for _, p in entries.values()))
        print("Output:", out_path.resolve())
        print("Size:", out_path.stat().st_size)
