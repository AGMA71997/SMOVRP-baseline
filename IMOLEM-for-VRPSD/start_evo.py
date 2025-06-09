import os
import multiprocessing

CPUcore = os.cpu_count()

command = []

pre = 'python main.py evo'

modes = ['lem']
datasets = [str(x) for x in range(20)]
MOmodes = ['DR']
problem_size = '50'
for mode in modes:
    for dataset in datasets:
        for MOmode in MOmodes:
            command.append(pre + ' ' + mode + ' ' + str(dataset) + ' ' + MOmode + ' '+ problem_size)

'''for cmd in command:
    print(cmd)
exit()'''

if __name__ == '__main__':
    p = multiprocessing.Pool(CPUcore)
    for cmd in command:
        p.apply_async(os.system, (cmd,))
    p.close()
    p.join()
