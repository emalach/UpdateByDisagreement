import glob
import sys
from gender import getGenders

indir = sys.argv[1]
infile = sys.argv[2]
cur_names = set()
with open(infile) as f:
    for line in f:
        cur_names.add(line.split()[0].strip())

dirlist = glob.glob('%s/*' % indir)
namelist = [s.split('/')[-1].split('_')[0] for s in dirlist]
names = list(set(namelist)-cur_names)
gender_dict = dict.fromkeys(names, None)
for i in range(0, len(names), 10):
    g = getGenders(names[i:i+10])
    for name, gender in zip(names[i:i+10], g):
        gender_dict[name] = gender
        print name, gender

