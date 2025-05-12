import argparse
import random
import numpy as np
import torch


class Evo_param:
    def __init__(self, size, maxiter, N=10, trace=False, MOmode='DRV', spec_init=True, spec_inst=True, no_tree=False):
        self.size = size
        self.maxiter = maxiter
        self.N = N
        self.trace = trace
        self.MOmode = MOmode
        self.spec_init = spec_init
        self.spec_inst = spec_inst
        self.no_tree = no_tree


class Problem:
    def __init__(self, problem_name, problem_size, capacity, *problem_data):
        self.problem_name = problem_name
        self.read_data_from_tensor(problem_size, capacity, *problem_data)

    def read_data_from_tensor(self, problem_size, *problem_data):
        node_demand, time_windows, service_times, travel_times = problem_data
        self.capacity = 1
        self.travel_times = travel_times
        self.time_bound = 1
        self.customers = []
        for n in range(problem_size + 1):
            if n == 0:
                cus = Customer(0, 0, 0.2, 0, [0, 1])
            else:
                cus = Customer(n, node_demand[n - 1], 0.2, service_times[n - 1], time_windows[n - 1])
            self.customers.append(cus)


class Customer:
    def __init__(self, id, demand, standard_deviation, servicetime, time_window):
        self.id = id
        self.demand = demand
        self.standard_deviation = standard_deviation
        self.servicetime = servicetime
        self.time_window = time_window
        self.actual_tt = {}

    def __repr__(self):
        return "customer {} at {}".format(self.id, id(self))

    def __str__(self):
        return "{}".format(self.id)

    def generate_actual_tts(self, N, other, travel_times):
        mean_tt = travel_times[other.id, self.id]
        self.actual_tt[other.id] = np.random.normal(mean_tt, self.standard_deviation, N)

    def get_distance(self, other, travel_times):
        return travel_times[other.id, self.id]


class Route:
    def __init__(self, customer_list, travel_times):
        self.customer_list = customer_list[:]
        self.set_mean_tt(travel_times)

    def __getitem__(self, i):
        return self.customer_list[i]

    def copy(self, travel_times):
        # 这里使用copy.deepcopy会导致客户内容也被复制
        new_route = type(self)(self.customer_list, travel_times)  # list没有复制是因为构造时会复制，复制到客户引用即可，不必复制客户内容
        return new_route

    def rand_seg_copy(self, travel_times):
        while True:
            start = random.randint(1, len(self.customer_list) - 2)
            end = random.randint(start + 1, len(self.customer_list) - 1)
            if start == 1 and end == len(self.customer_list) - 1:
                continue
            break
        new_route_customerlist = [self.customer_list[0]] + self.customer_list[start:end] + [self.customer_list[0]]
        new_route = type(self)(new_route_customerlist, travel_times)
        return new_route

    def set_mean_tt(self, travel_times):
        self.actual_tt_list = []
        other = None
        for customer in self.customer_list:
            if other is not None:
                self.actual_tt_list.append(customer.get_distance(other, travel_times))
                other = customer

    def set_one_actual_tt(self, i):
        self.actual_tt_list = []
        other = None
        for customer in self.customer_list:
            if other is not None:
                self.actual_tt_list.append(customer.actual_tt[other.id][i])
            other = customer

    def resource_consume(self, problem):
        sum_time = 0
        make_span = 0
        travel_time = 0
        remain_goods = problem.capacity
        travel_times = problem.travel_times
        now = 0
        goto = 1
        while goto < len(self.customer_list):
            ETA = sum_time + self.customer_list[goto].get_distance(self.customer_list[now], travel_times)
            ETA = max(ETA, self.customer_list[goto].time_window[0])
            if ETA < self.customer_list[goto].time_window[1] \
                    and self.customer_list[goto].demand < remain_goods:
                sum_time = ETA
                sum_time += self.customer_list[goto].servicetime
                remain_goods -= self.customer_list[goto].demand
                travel_time += self.customer_list[goto].get_distance(self.customer_list[now], travel_times)
                now = goto
                goto += 1
            else:
                remain_goods = problem.capacity
                if sum_time > make_span:
                    make_span = sum_time
                sum_time = 0
                now = 0
                # goto不变，因为需要回去继续服务
        return make_span, travel_time

    def find_customer(self, cus_ids):
        for cus in self.customer_list:
            if cus.id in cus_ids:
                return cus
        return None

    def random_shuffle(self):
        tmp = self.customer_list[1:-1]
        random.shuffle(tmp)
        self.customer_list = [self.customer_list[0]] + tmp + [self.customer_list[0]]


class Plan:
    def __init__(self, routes, avg_travel_times=None, avg_makespan=None):
        self.routes = routes[:]
        self.avg_travel_times = avg_travel_times
        self.avg_makespan = avg_makespan

    def copy(self):
        return type(self)([route.copy() for route in self.routes], self.avg_travel_times,
                          self.avg_makespan)

    def __getitem__(self, i):
        return self.routes[i]

    def arrange(self):
        self.routes = sorted(self.routes, key=lambda route: route.customer_list[1].id)

    def arrange_dis(self):
        self.routes = sorted(self.routes, key=lambda route: route.customer_list[1].dis_index, reverse=True)

    def equal(self, other):
        if len(self.routes) != len(other.routes):
            return False
        self.arrange()
        other.arrange()
        for self_route, other_route in zip(self.routes, other.routes):
            for sel_cus, oth_cus in zip(self_route.customer_list, other_route.customer_list):
                if sel_cus.id != oth_cus.id:
                    return False
        return True

    def equal_objective(self, other):
        if len(self.routes) == len(other.routes) and self.avg_makespan == other.avg_makespan \
                and self.avg_travel_times == other.avg_travel_times:
            return True
        else:
            return False

    def local_search_exploitation_SPS(self, problem):
        for route in self.routes:
            route.set_mean_tt(problem.travel_times)
        for index, route in enumerate(self.routes):
            old_makespan, old_travel_time = route.resource_consume(problem)
            new_route_customer_list = route.customer_list[1:-1]
            new_route_customer_list = sorted(new_route_customer_list,
                                             key=lambda customer: customer.get_distance(route.customer_list[0],
                                                                                        problem.travel_times),
                                             reverse=True)
            new_route_customer_list = [route.customer_list[0]] + new_route_customer_list + [route.customer_list[0]]
            new_route = Route(new_route_customer_list, problem.travel_times)
            new_makespan, new_travel_time = new_route.resource_consume(problem)
            if new_travel_time < old_travel_time:
                self.routes[index] = new_route

    def local_search_exploitation_WDS(self, problem):
        for route in self.routes:
            route.set_mean_tt(problem.travel_times)
        for index, route in enumerate(self.routes):
            _, old_travel_time = route.resource_consume(problem)
            new_route_customer_list = route.customer_list[:]
            new_route_customer_list.reverse()
            new_route = Route(new_route_customer_list, problem.travel_times)
            _, new_travel_time = new_route.resource_consume(problem)
            if new_travel_time < old_travel_time:
                self.routes[index] = new_route

    def __remove_duplicated_customers(self, base):
        del_cus_id = []
        for customer in self.routes[base].customer_list[1:-1]:
            del_cus_id.append(customer.id)
        for route in self.routes[:-1]:
            while route.find_customer(del_cus_id):
                route.customer_list.remove(route.find_customer(del_cus_id))
                if len(route.customer_list) == 2:
                    self.routes.remove(route)

    def route_crossover(self, other_plan, random_shuffling_rate):
        if len(self.routes) != 1:
            r1 = self.routes[random.randint(0, len(self.routes) - 1)].copy()
        else:
            r1 = self.routes[0].rand_seg_copy()
        if len(other_plan.routes) != 1:
            r2 = other_plan.routes[random.randint(0, len(other_plan.routes) - 1)].copy()
        else:
            r2 = other_plan.routes[0].rand_seg_copy()
        self.routes.append(r2)
        other_plan.routes.append(r1)
        self.__remove_duplicated_customers(-1)
        other_plan.__remove_duplicated_customers(-1)

        for route in self.routes:
            if len(route.customer_list) == 2:
                self.routes.remove(route)
        for route in other_plan.routes:
            if len(route.customer_list) == 2:
                other_plan.routes.remove(route)

        for i in range(len(self.routes) - 1):
            if random.random() < random_shuffling_rate:
                self.routes[i].random_shuffle()
        for i in range(len(other_plan.routes) - 1):
            if random.random() < random_shuffling_rate:
                other_plan.routes[i].random_shuffle()

    def __partial_swap(self):
        if len(self.routes) == 1:
            return
        swap_route_1 = random.randint(0, len(self.routes) - 1)
        swap_route_2 = random.randint(0, len(self.routes) - 1)
        while swap_route_2 == swap_route_1:
            swap_route_2 = random.randint(0, len(self.routes) - 1)

        route_1_part_start = random.randint(1, len(self.routes[swap_route_1].customer_list) - 2)
        route_1_part_end = random.randint(2, len(self.routes[swap_route_1].customer_list) - 1)
        while route_1_part_end <= route_1_part_start:
            route_1_part_end = random.randint(2, len(self.routes[swap_route_1].customer_list) - 1)
        route_2_part_start = random.randint(1, len(self.routes[swap_route_2].customer_list) - 2)
        route_2_part_end = random.randint(2, len(self.routes[swap_route_2].customer_list) - 1)
        while route_2_part_end <= route_2_part_start:
            route_2_part_end = random.randint(2, len(self.routes[swap_route_2].customer_list) - 1)

        tmp = self.routes[swap_route_1].customer_list[route_1_part_start:route_1_part_end]
        self.routes[swap_route_1].customer_list[route_1_part_start:route_1_part_end] = self.routes[
                                                                                           swap_route_2].customer_list[
                                                                                       route_2_part_start:route_2_part_end]
        self.routes[swap_route_2].customer_list[route_2_part_start:route_2_part_end] = tmp

    def __merge_shortest_route(self, problem):
        if len(self.routes) == 1:
            return
        for route in self.routes:
            route.set_mean_tt(problem.travel_times)
        sorted_routes = sorted(self.routes, key=lambda route: route.resource_consume(problem)[1])
        sorted_routes[0].customer_list.pop(-1)
        sorted_routes[1].customer_list.pop(0)
        sorted_routes[0].customer_list.extend(sorted_routes[1].customer_list)
        self.routes.remove(sorted_routes[1])

    def __split_longest_route(self, problem):
        for route in self.routes:
            route.set_mean_tt(problem.travel_times)
        sorted_routes = sorted(self.routes, key=lambda route: route.resource_consume(problem)[1], reverse=True)
        if len(sorted_routes[0].customer_list) == 3:
            return
        break_first_end = random.randint(2, len(sorted_routes[0].customer_list) - 2)
        second_cus_list = [sorted_routes[0].customer_list[0]] + sorted_routes[0].customer_list[break_first_end:]
        self.routes.append(Route(second_cus_list, problem.travel_times))
        sorted_routes[0].customer_list[break_first_end:-1] = []

    def mutation(self, problem, mutation_rate, elastic_rate, squeeze_rate, shuffle_rate):
        if random.random() < mutation_rate:
            if random.random() < elastic_rate:
                self.__partial_swap()
            elif random.random() < squeeze_rate:
                self.__merge_shortest_route(problem)
            else:
                self.__split_longest_route(problem)
            for route in self.routes:
                if random.random() < shuffle_rate:
                    route.random_shuffle()

    def RSM(self, N, problem):
        sum_makespan = 0
        sum_tt = 0
        for i in range(N):
            for route in self.routes:
                route.set_one_actual_tt(i)
                makespan, tt = route.resource_consume(problem)
                sum_makespan += makespan
                sum_tt += tt
        self.avg_makespan = sum_makespan / N
        self.avg_travel_times = sum_tt / N

    def get_objective(self):
        return self.avg_makespan, self.avg_travel_times, len(self.routes)  # DRV

    def cal_difference(self, other_plan):
        objective1 = np.array(self.get_objective())
        objective2 = np.array(other_plan.get_objective())
        diff_vec = objective1 - objective2
        return np.linalg.norm(diff_vec)


class VectorPlan:
    # 气泡优先级编码
    def __init__(self, customers, plan=None, vector=None):
        if plan != None:
            plan.arrange()
            route_num = len(plan.routes)
            max_priority = len(customers) - 1 + route_num - 1
            self.vector = [0] * (len(customers) - 1)
            for route in plan.routes:
                for cus in route.customer_list:
                    if cus.id != 0:
                        self.vector[cus.id - 1] = max_priority
                        max_priority -= 1
                max_priority -= 1

        else:
            if vector != None:
                self.vector = vector

    def backto_plan(self, customers, travel_times):
        max_priority = max(self.vector)
        routes = []
        customer_list = [customers[0]]
        while max_priority >= 0:
            try:
                cus_id = self.vector.index(max_priority) + 1
                customer_list.append(customers[cus_id])
                max_priority -= 1
            except(ValueError):
                max_priority -= 1
                customer_list.append(customers[0])
                if len(customer_list) > 2:
                    routes.append(Route(customer_list, travel_times))
                customer_list = [customers[0]]
        plan = Plan(routes)

        return plan


def main():
    parser = argparse.ArgumentParser(
        description='Train Active Search on Instance')
    parser.add_argument('--problem_size', type=int, default=50, help='Number of customers')
    parser.add_argument('--index', type=int, default=1, help='Number of customers')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    run_args = parser.parse_args()

    import os
    cwd = os.getcwd()
    data_path = os.path.dirname(cwd) + "/data/"

    depot_xy, node_xy, node_demand, time_windows, service_times, travel_times = torch.load(
        data_path + 'problem data ' + str(run_args.problem_size),
        map_location=device)
    node_demand = node_demand[run_args.index]
    time_windows = time_windows[run_args.index]
    service_times = service_times[run_args.index]
    travel_times = travel_times[run_args.index]
    problem_name = "instance_no_" + str(run_args.index)
    problem = Problem(problem_name, run_args.problem_size, node_demand, time_windows,
                      service_times, travel_times)
    print(problem.problem_name)


if __name__ == '__main__':
    main()
