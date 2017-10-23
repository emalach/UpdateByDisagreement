import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

# TODO: unique snapshots

dirbase = sys.argv[1]

snapshots = range(0,30000,200) + [29999]

max_ref = []
max_1 = []
max_2 = []

all_ref = np.zeros((len(snapshots), 3))
all_1 = np.zeros((len(snapshots), 3))
all_2 = np.zeros((len(snapshots), 3))

for i in range(3):
    dirname = '%s%d' % (dirbase, i)
    dual_run_num = glob.glob(dirname + '/*f#2.csv')[0].split('run_run-')[1].split(',')[0]
    ref_run_num = [f.split('run_run-')[1].split(',')[0] for f in glob.glob(dirname + '/*f.csv') if dual_run_num not in f][0]

    # get file names
    ref_f = glob.glob(dirname + '/*' + ref_run_num + '*f.csv')[0]
    ref_m = glob.glob(dirname + '/*' + ref_run_num + '*m.csv')[0]
    dual_f_1 = glob.glob(dirname + '/*' + dual_run_num + '*f.csv')[0]
    dual_m_1 = glob.glob(dirname + '/*' + dual_run_num + '*m.csv')[0]
    dual_f_2 = glob.glob(dirname + '/*' + dual_run_num + '*f#2.csv')[0]
    dual_m_2 = glob.glob(dirname + '/*' + dual_run_num + '*m#2.csv')[0]

    # parse files
    f_ref = np.genfromtxt(ref_f, delimiter=',')
    m_ref = np.genfromtxt(ref_m, delimiter=',')
    f_1 = np.genfromtxt(dual_f_1, delimiter=',')
    m_1 = np.genfromtxt(dual_m_1, delimiter=',')
    f_2 = np.genfromtxt(dual_f_2, delimiter=',')
    m_2 = np.genfromtxt(dual_m_2, delimiter=',')

    # remove initial nan
    f_ref = f_ref[1:,:]
    m_ref = m_ref[1:,:]
    f_1 = f_1[1:,:]
    m_1 = m_1[1:,:]
    f_2 = f_2[1:,:]
    m_2 = m_2[1:,:]

    # confirm correctness
    if np.sum(f_ref[:,1] != m_ref[:,1]) > 0 or np.sum(f_1[:,1] != m_1[:,1]) > 0 or np.sum(f_2[:,1] != m_2[:,1]) > 0:
        print('PROBLEM WITH SNAPSHOTS SYNCHRONIZATION!')
        exit(-1)

    # get unique snapshots
    _, idx = np.unique(f_ref[:,1], return_index = True)
    f_ref = f_ref[idx,:]
    m_ref = m_ref[idx,:]
    a, idx = np.unique(f_1[:,1], return_index = True)
    f_1 = f_1[idx,:]
    m_1 = m_1[idx,:]
    _, idx = np.unique(f_2[:,1], return_index = True)
    f_2 = f_2[idx,:]
    m_2 = m_2[idx,:]

    mean_ref = np.mean(np.column_stack((f_ref[:,2], m_ref[:,2])), axis=1)
    mean_1 = np.mean(np.column_stack((f_1[:,2], m_1[:,2])), axis=1)
    mean_2 = np.mean(np.column_stack((f_2[:,2], m_2[:,2])), axis=1)

    # better should be first
    if np.max(mean_2) > np.max(mean_1):
        tmp = mean_1
        mean_1 = mean_2
        mean_2 = tmp

    # fix holes in snapshots... 
    for j,snap in enumerate(snapshots):
        if snap in f_ref[:,1]:
            all_ref[j,i] = mean_ref[f_ref[:,1]==snap]
        else:
            all_ref[j,i] = all_ref[j-1,i]

        if snap in f_1[:,1]:
            all_1[j,i] = mean_1[f_1[:,1]==snap]
        else:
            all_1[j,i] = all_1[j-1,i]

        if snap in f_2[:,1]:
            all_2[j,i] = mean_2[f_2[:,1]==snap]
        else:
            all_2[j,i] = all_2[j-1,i]


    max_ref.append(np.max(mean_ref))
    max_1.append(np.max(mean_1))
    max_2.append(np.max(mean_2))

error_ref = np.std(all_ref[25:], axis=1)
y_ref = np.mean(all_ref[25:], axis=1)
error_1 = np.std(all_1[25:], axis=1)
y_1 = np.mean(all_1[25:], axis=1)
plt.plot(snapshots[25:], y_ref)
plt.plot(snapshots[25:], y_1)
plt.fill_between(snapshots[25:], y_ref-error_ref, y_ref+error_ref, facecolor='c', linewidth=0)
plt.fill_between(snapshots[25:], y_1-error_1, y_1+error_1, facecolor='#7EFF99', linewidth=0)
plt.show()


print '%.3f +- %.3f' % (np.mean(max_ref), np.std(max_ref))
print '%.3f +- %.3f' % (np.mean(max_1), np.std(max_1))
print '%.3f +- %.3f' % (np.mean(max_2), np.std(max_2))



