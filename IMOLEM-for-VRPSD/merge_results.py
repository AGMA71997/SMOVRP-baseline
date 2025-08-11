import pickle
import os
from vrp import util

def merge_results(directory,inst_size):
    results = []
    for file_index in range(20):
        file_name = "instance_no_"+str(file_index)
        print(file_name)
        file_path = directory+'/'+file_name
        file_path = file_path+'/'+'lem_DRV_DR_'+str(inst_size)+'_population.pickle'
        pickle_in = open(file_path, 'rb')
        instance_Q = pickle.load(pickle_in)
        non_dom_Q = util.pareto_first(instance_Q)
        inst_results = []
        for plan in non_dom_Q:
            inst_results.append((plan.avg_travel_times,plan.avg_makespan))
        results.append(inst_results)

    print(len(results))
    print(results)

    results_file = open('results '+str(inst_size), 'wb')
    pickle.dump(results, results_file)
    results_file.close()

def main():
    import sys
    sys.path.insert(0, "result/..")  # for utils
    instance_size = 50
    directory = "result/"+str(instance_size)
    merge_results(directory,instance_size)


if __name__ == '__main__':
    main()