import random
import copy
import numpy


def relatedness_removal(problem, current, val_matrix,
                        nr_nodes_to_remove=None, prob=5):
    destroyed_solution = copy.deepcopy(current)
    visited_customers = [customer for route in destroyed_solution for customer in route
                         if customer != 0]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(problem.num_customers)

    node_to_remove = random.choice(visited_customers)
    for route in destroyed_solution:
        while node_to_remove in route:
            route.remove(node_to_remove)
            visited_customers.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        related_nodes = []
        normalized_distances = scaled(val_matrix[node_to_remove, :])
        route_node_to_remove = [route for route in current if node_to_remove in route][0]
        for route in destroyed_solution:
            for node in route:
                if node != 0:
                    if node in route_node_to_remove:
                        related_nodes.append((node, normalized_distances[node]))
                    else:
                        related_nodes.append((node, normalized_distances[node] + 1))

        if random.random() < 1 / prob:
            node_to_remove = random.choice(visited_customers)
        else:
            node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
        for route in destroyed_solution:
            while node_to_remove in route:
                route.remove(node_to_remove)
                visited_customers.remove(node_to_remove)
    destroyed_solution = [route for route in destroyed_solution if (route != []
                                                                    and route != [0, 0])]

    return destroyed_solution

def get_regret_single_insertion(problem, routes, customer,val_matrix):
    relevant_insertions = {}
    for route_idx in range(len(routes)):
        for i in range(len(routes[route_idx])+1):
            dict_key = (customer, i, tuple(routes[route_idx]))
            if dict_key not in self.insertions:
                updated_route = routes[route_idx][:i] + [customer] + routes[route_idx][i:]
                if check_route_feasibility(updated_route, problem.time_matrix, problem.time_windows,
                                        problem.service_times, problem.demands,
                                        problem.vehicle_capacity):
                    if i == 0:
                        cost_difference = val_matrix[0, updated_route[0]] + val_matrix[
                            updated_route[0], updated_route[1]] - val_matrix[0, updated_route[1]]
                    elif i == len(routes[route_idx]):
                        cost_difference = val_matrix[updated_route[-1], 0] + val_matrix[
                            updated_route[i - 1], updated_route[i]] - val_matrix[updated_route[i - 1], 0]
                    else:
                        cost_difference = val_matrix[updated_route[i - 1], updated_route[i]] + \
                                          val_matrix[updated_route[i], updated_route[i + 1]] - \
                                          val_matrix[updated_route[i - 1], updated_route[i + 1]]

                    problem.insertions[dict_key] = cost_difference
                    relevant_insertions[dict_key] = cost_difference

                else:
                    self.insertions[dict_key] = False
                    relevant_insertions[dict_key] = False

            else:
                relevant_insertions[dict_key] = self.insertions[dict_key]

    relevant_insertions = {key: relevant_insertions[key] for key in relevant_insertions if
                           relevant_insertions[key] != False}

    if len(relevant_insertions) == 1:
        best_insertion = min(relevant_insertions, key=relevant_insertions.get)
        return best_insertion, 0

    elif len(relevant_insertions) > 1:
        best_insertion = min(relevant_insertions, key=relevant_insertions.get)

        if len(set(relevant_insertions.values())) == 1:  # when all options are of equal value:
            regret = 0
        else:
            regret = sorted(list(relevant_insertions.values()))[1] - min(relevant_insertions.values())
        return best_insertion, regret
    else:
        # no insertions possible for this customer
        return -1, -1

def regret_insertion(problem,current, val_matrix, prob=1.5):
    visited_customers = [customer for route in current for customer in route]
    all_customers = set(range(1, problem.num_customers + 1))
    unvisited_customers = all_customers - set(visited_customers)

    repaired = copy.deepcopy(current)
    while unvisited_customers:
        insertion_options = {}
        for customer in unvisited_customers:
            best_insertion, regret = get_regret_single_insertion(problem, repaired, customer,
                                                                      val_matrix)
            if best_insertion != -1:
                insertion_options[best_insertion] = regret

        if not insertion_options:
            repaired.append([0,random.choice(list(unvisited_customers)),0])
        else:
            insertion_option = 0
            while random.random() < 1 / prob and insertion_option < len(insertion_options) - 1:
                insertion_option += 1

            insertion_operation = sorted(insertion_options, reverse=True)[insertion_option]
            customer = insertion_operation[0]
            customer_index = insertion_operation[1]
            route = list(insertion_operation[2])
            route_index = repaired.index(route)
            repaired[route_index].insert(customer_index, customer)

        visited_customers = [customer for route in repaired for customer in route]
        unvisited_customers = all_customers - set(visited_customers)
    return repaired


def determine_nr_nodes_to_remove(nb_customers, omega_bar_minus=5, omega_minus=0.1, omega_bar_plus=30, omega_plus=0.4):
    n_plus = min(omega_bar_plus, omega_plus * nb_customers)
    n_minus = min(n_plus, max(omega_bar_minus, omega_minus * nb_customers))
    r = random.randint(round(n_minus), round(n_plus))
    return r

def check_route_feasibility(route, time_matrix, time_windows, service_times, demands_data, truck_capacity):
    if len(route) < 3 or route[0] != 0 or route[-1] != 0:
        return False

    current_time = max(time_matrix[0, route[1]], time_windows[route[1], 0])
    total_capacity = 0

    for i in range(1, len(route)):
        if round(current_time, 3) > time_windows[route[i], 1]:
            #print("Time Window violated")
            return False
        current_time += service_times[route[i]]
        total_capacity += demands_data[route[i]]
        if round(total_capacity, 3) > truck_capacity:
            #print("Truck Capacity Violated")
            return False
        if i < len(route) - 1:
            # travel to next node
            current_time += time_matrix[route[i], route[i + 1]]
            current_time = max(current_time, time_windows[route[i + 1], 0])
    return True

def scaled(matrix):
    max_val = numpy.max(matrix)
    min_val = numpy.min(matrix)
    return (matrix - min_val) / (max_val - min_val)