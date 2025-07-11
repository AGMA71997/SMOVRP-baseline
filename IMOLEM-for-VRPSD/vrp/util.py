import random
import numpy as np
import geatpy as ea
from sklearn import tree

from .vrpclass import Route, Plan, VectorPlan


def build_first_plan(problem, n_sols=1):
    for customer in problem.customers:
        assert problem.capacity > customer.demand and problem.time_bound > customer.servicetime

    init_P = []
    for sol in range(n_sols):
        routes = []
        load = 0
        time = 0
        travel_times = problem.travel_times
        route_cuslist = [problem.customers[0]]
        i = 0
        select_sequence = list(range(1, len(problem.customers)))
        random.shuffle(select_sequence)
        while i < len(select_sequence):

            load += problem.customers[select_sequence[i]].demand
            time += route_cuslist[-1].get_distance(problem.customers[select_sequence[i]], travel_times)
            time = max(time, problem.customers[select_sequence[i]].time_window[0])
            if load < problem.capacity and time < problem.customers[select_sequence[i]].time_window[1]:
                time += problem.customers[select_sequence[i]].servicetime
                route_cuslist.append(problem.customers[select_sequence[i]])
                i += 1
            else:
                route_cuslist.append(problem.customers[0])
                routes.append(Route(route_cuslist, travel_times))
                route_cuslist = [problem.customers[0]]
                load = 0
                time = 0
        if len(route_cuslist) > 1:
            route_cuslist.append(problem.customers[0])
            routes.append(Route(route_cuslist, travel_times))
        init_P.append(Plan(routes))
    return init_P


# 优先级编码实例化
def instantiating(problem, size, max_route, tree):
    Q = []
    rules = explain_tree(tree, problem.customers)
    if '1' not in rules:
        return build_first_plan(problem)

    positive_rules = []
    for positive_rule in rules['1']:
        if instantiating_sub(problem, max_route, positive_rule) != None:
            positive_rules.append(positive_rule)

    if len(positive_rules) == 0:
        return []

    avg_inst_num = int(size / len(positive_rules))
    if avg_inst_num < 1:
        avg_inst_num = 1

    for positive_rule in positive_rules:
        for _ in range(avg_inst_num):
            ret = instantiating_sub(problem, max_route, positive_rule)
            Q.append(ret)

    while len(Q) < size:
        positive_rule = random.choice(positive_rules)
        ret = instantiating_sub(problem, max_route, positive_rule)
        Q.append(ret)
    while None in Q:
        Q.remove(None)
    return Q


def instantiating_sub(problem, max_route, positive_rule):
    # cur_max_route = random.randint(1, max_route)
    max_priority = len(problem.customers) - 1 + max_route - 1
    select = list(range(1, max_priority + 1))
    assigned = []
    vector = [None] * (len(problem.customers) - 1)

    positive_rule = [rule.split() for rule in positive_rule]
    for rule in positive_rule:
        rule[0] = int(rule[0])
        rule[2] = float(rule[2])

    rule_fesible_region = {}
    rule_freedom = {}
    for rule in positive_rule:
        feasible_region = set([sel for sel in select if eval('sel {} {}'.format(rule[1], rule[2]))])
        rule_fesible_region[rule[0]] = feasible_region
        rule_freedom[rule[0]] = len(feasible_region)
    rule_freedom = sorted(rule_freedom.items(), key=lambda kv: (kv[1], kv[0]))

    for rule_index, _ in rule_freedom:
        if len(rule_fesible_region[rule_index]) == 0:
            return None
        vector[rule_index] = random.choice(list(rule_fesible_region[rule_index]))
        for k in rule_fesible_region:
            rule_fesible_region[k] -= {vector[rule_index]}
        select.remove(vector[rule_index])
        assigned.append(rule_index)

    for i in range(len(vector)):
        if i not in assigned:
            vector[i] = random.choice(select)
            select.remove(vector[i])

    assert None not in vector
    vec_plan = VectorPlan(problem.customers, vector=vector).backto_plan(problem.customers, problem.travel_times)
    return vec_plan


def explain_tree(clf, customers):
    class_rules_dict = {}
    tree_str = tree.export_text(clf, feature_names=[i for i in range(0, len(customers) - 1)])

    tree_str_rows = tree_str.split('\n')[:-1]  # 最后一个是空行，扔掉
    branch_count = [line.count('|') for line in tree_str_rows]
    for line_no, line in enumerate(tree_str_rows):
        if 'class:' in line:
            class_ = line[line.rfind(' ') + 1:]
            if class_ not in class_rules_dict:
                class_rules_dict[class_] = []
            rule = []
            for count in range(branch_count[line_no] - 1, 0, -1):
                rule_node_no = list_reverse_index(branch_count[:line_no], count)
                rule_node_str = tree_str_rows[rule_node_no]
                rule_node_str = rule_node_str[rule_node_str.find('---') + 4:]
                rule.append(rule_node_str)
            rule.reverse()
            class_rules_dict[class_].append(rule)
    return class_rules_dict


def list_reverse_index(list_, target):
    rev_list = list_[:]
    rev_list.reverse()
    return len(list_) - 1 - rev_list.index(target)


def deduplicate(P):
    newP = []
    for plan in P:
        have_duplicate = False
        for exist_plan in newP:
            if plan.equal(exist_plan):
                have_duplicate = True
                break
        if not have_duplicate:
            newP.append(plan)
    return newP


def deduplicate_objective(P):
    newP = []
    for plan in P:
        have_duplicate = False
        for exist_plan in newP:
            if plan.equal_objective(exist_plan):
                have_duplicate = True
                break
        if not have_duplicate:
            newP.append(plan)
    return newP


def pareto_sort(P, needNum=None, needLevel=None, MOmode='DRV'):
    objv = []
    for plan in P:
        if MOmode == 'DRV':
            objv.append(plan.get_objective())
        elif MOmode == 'DR':
            objv.append([plan.avg_makespan, plan.avg_travel_times])
        elif MOmode == 'DV':
            objv.append([plan.avg_makespan, len(plan.routes)])
        elif MOmode == 'RV':
            objv.append([plan.avg_travel_times, len(plan.routes)])
        else:
            assert 'error: wrong MOmode'
    objv = np.array(objv)
    levels, criLevel = ea.ndsortESS(objv, needNum, needLevel)
    dis = ea.crowdis(objv, levels)
    sortP = []
    for lv in range(1, criLevel):
        indexs = np.argwhere(levels == lv)
        indexs_sorted = sorted(indexs, key=lambda x: dis[x[0]], reverse=True)
        for i in indexs_sorted:
            sortP.append(P[i[0]])
    indexs = np.argwhere(levels == criLevel)
    indexs_sorted = sorted(indexs, key=lambda x: dis[x[0]], reverse=True)
    for i in indexs_sorted:
        if len(sortP) < needNum:
            sortP.append(P[i[0]])
    return sortP


def target_sort(P, size, target):
    if target == 'D':
        sortP = sorted(P, key=lambda plan: plan.avg_makespan)
    elif target == 'R':
        sortP = sorted(P, key=lambda plan: plan.avg_travel_times)
    elif target == 'V':
        sortP = sorted(P, key=lambda plan: len(plan.routes))
    else:
        assert ('error: wrong target')
    if len(P) > size:
        sortP = sortP[:size]
    return sortP


def indicator_HV(P):
    objv = []
    for plan in P:
        objv.append(plan.get_objective())
    objv = np.array(objv)
    hv = ea.indicator.HV(objv)
    return hv


def indicator_Spacing(P):
    if len(P) <= 1:
        return 0
    objv = []
    for plan in P:
        objv.append(plan.get_objective())
    objv = np.array(objv)
    sc = ea.indicator.Spacing(objv)
    return sc


def cal_result(result_population, N, problem):
    actual_tt_backup = {}
    for cus in problem.customers:
        actual_tt_backup[cus.id] = cus.actual_tt.copy()
        for key in cus.actual_tt:
            other = problem.customers[key]
            cus.generate_actual_tt(N, other, problem.travel_times)
    for plan in result_population:
        plan.RSM(N, problem)
    for cus in problem.customers:
        cus.actual_tt = actual_tt_backup[cus.id]

    sum_routes = 0
    sum_makespan = 0
    sum_travel_time = 0
    for plan in result_population:
        sum_routes += len(plan.routes)
        sum_makespan += plan.avg_makespan
        sum_travel_time += plan.avg_travel_times
    all_avg_routes = sum_routes / len(result_population)
    all_avg_makespan = sum_makespan / len(result_population)
    all_avg_tt = sum_travel_time / len(result_population)

    retstr = ''
    retstr += 'sum plan number= ' + str(len(result_population)) + '\n'
    retstr += 'avg_routes: {}\navg_distance: {}\navg_pay: {}'.format(all_avg_routes, all_avg_makespan,
                                                                     all_avg_tt) + '\n'
    hv = indicator_HV(result_population)
    spacing = indicator_Spacing(result_population)
    retstr += 'hv: ' + str(hv) + '\n' + 'spacing: ' + str(spacing)
    return (all_avg_routes, all_avg_makespan, all_avg_tt, hv, spacing), retstr


def show_result(result_population, *argl):
    sum_routes = 0
    sum_makespan = 0
    sum_tt = 0
    for plan in result_population:
        sum_routes += len(plan.routes)
        sum_makespan += plan.avg_makespan
        sum_tt += plan.avg_travel_times
    all_avg_routes = sum_routes / len(result_population)
    all_avg_makespan = sum_makespan / len(result_population)
    all_avg_tt = sum_tt / len(result_population)

    retstr = ''
    retstr += 'sum plan number= ' + str(len(result_population)) + '\n'
    retstr += 'avg_routes: {}\navg_makespan: {}\navg_travel_time: {}'.format(all_avg_routes, all_avg_makespan,
                                                                     all_avg_tt) + '\n'
    hv = indicator_HV(result_population)
    spacing = indicator_Spacing(result_population)
    retstr += 'hv: ' + str(hv) + '\n' + 'spacing: ' + str(spacing)
    return (all_avg_routes, all_avg_makespan, all_avg_tt, hv, spacing), retstr


def customers_sort_by_distance(customers, problem):
    customers0 = customers[0]
    customers.sort(key=lambda cus: cus.get_distance(customers0, problem.travel_times))
    dis_index = 0
    for cus in customers:
        cus.dis_index = dis_index
        dis_index += 1


def check_plan_legal(P, customers):
    cus_id = {cus.id for cus in customers}
    for plan in P:
        plan_cus_id = set()
        cus_num = 0
        for route in plan.routes:
            for cus in route.customer_list:
                plan_cus_id.add(cus.id)
                cus_num += 1
            cus_num -= 2
        assert (cus_num == len(customers) - 1)
        assert (cus_id == plan_cus_id)
    print('check passed')


def pareto_first(P):
    objv = []
    for plan in P:
        objv.append(plan.get_objective())
    objv = np.array(objv)
    levels, _ = ea.ndsortESS(objv)
    Pfirst = []
    indexs = np.argwhere(levels == 1)
    for i in indexs:
        Pfirst.append(P[i[0]])
    return Pfirst


def pk(length, times):
    pk = []
    for pk_i in range(times):
        pk.append(list(range(length)))
        random.shuffle(pk[pk_i])
    pk_win = []
    for pk_i in range(times):
        pk_win.append([])
        for list_i in range(0, length - 1, 2):
            if pk[pk_i][list_i] < pk[pk_i][list_i + 1]:
                pk_win[pk_i].append(pk[pk_i][list_i])
            else:
                pk_win[pk_i].append(pk[pk_i][list_i + 1])

    new_pk_win = []
    for pk_win_one in pk_win:
        for i in pk_win_one:
            new_pk_win.append(i)
    random.shuffle(new_pk_win)
    # print(new_pk_win)
    return new_pk_win
