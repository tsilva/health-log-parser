import os
import re
from pathlib import Path

def extract_dates_from_dir(directory):
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})")
    dates = set()
    for fname in os.listdir(directory):
        m = date_pattern.match(fname)
        if m:
            dates.add(m.group(1))
    return dates

def main():
    dir1 = "./output/input/"
    dir2 = "/mnt/c/Users/engti/Desktop/health - labs - cristina/"
    dates1 = extract_dates_from_dir(dir1)
    dates2 = extract_dates_from_dir(dir2)
    only_in_dir2 = sorted(dates2 - dates1)
    print(f"Dates in {dir2} but not in {dir1}:")
    for d in only_in_dir2:
        print(d)

if __name__ == "__main__":
    import sys
    main()
