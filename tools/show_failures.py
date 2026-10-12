"""Print JUnit failures with the class and test they actually belong to (proper XML parse)."""
import glob
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/projects/laya-module/app/build/test-results/testDebugUnitTest"
total = failed = 0
for f in sorted(glob.glob(path + "/*.xml")):
    tree = ET.parse(f)
    for case in tree.getroot().iter("testcase"):
        total += 1
        failure = case.find("failure")
        if failure is None:
            continue
        failed += 1
        print("=" * 70)
        print("CLASS " + str(case.get("classname")))
        print("TEST  " + str(case.get("name")))
        print("-" * 70)
        print((failure.get("message") or "")[:2500])
        print()
print("total " + str(total) + ", failed " + str(failed))