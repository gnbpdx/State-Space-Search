import copy
import sys
import math
import random
import getopt

global num_nodes_visited
global backtracks
#For each node in the graph self.neighbors[node] is a list of the neighbors of that node
# For each node self.degree[node] is the degree of the node
class Graph():
    def __init__(self, filename):
        reader = list(self.read_dimacs(filename))
        self.neighbors = {}
        self.degree = {}
        for line in reader:
            if line != '\n':
                readline = line.split()
                if readline[0] == 'c':
                    continue
                if readline[0] == 'p':
                    self.num_nodes = int(readline[2])
                    for node in range(1, self.num_nodes + 1):
                        self.neighbors[node] = []
                if readline[0] == 'e':
                    self.insert_edge(int(readline[1]), int(readline[2]))
        for node in range(1, self.num_nodes + 1):
            self.degree[node] = len(self.neighbors[node])
        self.max_degree = max([self.degree[node] for node in range(1, self.num_nodes + 1)])
    def insert_edge(self, node_1, node_2):
        if not node_2 in self.neighbors[node_1]:
            self.neighbors[node_1].append(node_2)
        if not node_1 in self.neighbors[node_2]:
            self.neighbors[node_2].append(node_1)

    def read_dimacs(self, filename):
        with open(filename) as f:
            for line in f:
                yield line
# For each node in the graph self.colors[node] are the possible colors that node can be.
# That is all the colors for which no neighbor to that node has been selected to be that color.
# self.variables_left is a list of nodes for which colors still have to be selected for.
# For each color self.values[color] is a list of all nodes that have been selected for that color. 
class DFS_State():
    def __init__(self, graph, num_colors):
        self.graph = graph
        self.num_colors = num_colors
        self.colors = {}
        for node in range(1, graph.num_nodes + 1):
            self.colors[node] = list(range(1, num_colors + 1))
        self.variables_left = list(range(1, graph.num_nodes + 1))
        self.removed_variables = []
        for node in graph.neighbors:
            if self.graph.degree[node] < num_colors:
                self.removed_variables.append(node)
                self.variables_left.remove(node)
        self.variables_left_copy = copy.deepcopy(self.variables_left)
        self.values = {}
        for color in range(1, num_colors + 1):
            self.values[color] = []

    def reset_state(self):
        for node in range(1, self.graph.num_nodes + 1):
            self.colors[node] = list(range(1, self.num_colors + 1))
        self.variables_left = copy.deepcopy(self.variables_left_copy)
        for color in range(1, self.num_colors + 1):
            self.values[color] = []

    # When we assign a color to a node, self.colors changes.
    # self.recent_changes keeps track of these changes so we can revert back to the old state
    def adjust_color_possibilities(self, node, value):
        self.recent_changes = {}
        node_changes = copy.deepcopy(self.colors[node])
        node_changes.remove(value)
        self.recent_changes[node] = node_changes
        for variable in self.graph.neighbors[node]:
            if value in self.colors[variable]:
                self.recent_changes[variable] = [value]
                self.colors[variable].remove(value)

    # Use recent_changes to revert back to the old state.
    # Note: Don't use self.recent_changes as we want the changes pushed onto the stack before recursing down.
    def reset_color_possibilities(self, recent_changes):
        for variable in recent_changes:
            for value in recent_changes[variable]:
                self.colors[variable].append(value)

    def finish_search(self):
        self.variables_left = self.removed_variables
        if self.variables_left:
            return self.Search(self.DFS_Variable_Heuristic(), self.DFS_Variable_Heuristic, self.DFS_Value_Heuristic, math.inf, 1, math.inf)
        return True

    def cost(self, variable, value):
        return sum([1 for neighbor in self.graph.neighbors[variable] if value in self.colors[neighbor]])
    # Value pruning removes colors that have not been assigned to any node if some color fails that has not been assigned to a node
    # This is a way of eliminating some combination of colors that are a rearrangement of colors we have already tried
    # Do not use value pruning if Search is not done as a true DFS (DDS/LDS)

    def Search(self, node, variable_heuristic, value_heuristic, discrepancies_left, depth, depth_limit, value_pruning=False, ilds=False):
        global num_nodes_visited
        global backtracks
        used_heuristic = False
        num_nodes_visited += 1
        #All variables have been assigned
        number_variables_left = len(self.variables_left)
        if number_variables_left == 0:
            return True
        
        #Some variable has no possible assignment (Forward Pruning)
        for variable in self.variables_left:
            if not self.colors[variable]:
                return False
        
        #Get list of possible values to assign variable sorted by order to try
        possible_values = value_heuristic(node)
        
        #Find next variable to use
        self.variables_left.remove(node)
        new_variable = None
        if self.variables_left:
            new_variable = variable_heuristic()

        #Iterate through possible values
        while(possible_values):
            if not used_heuristic or (discrepancies_left > 0 and depth <= depth_limit) or depth == depth_limit:
                value_try = possible_values[0]
                if not used_heuristic:
                    if depth == depth_limit or (ilds and number_variables_left <= discrepancies_left):
                        possible_values.remove(value_try)
                        used_heuristic = True
                        continue
                #Change neighbors possible values
                self.adjust_color_possibilities(node, value_try)
                #Get changes to neighbors values
                color_changes = copy.deepcopy(self.recent_changes)
                #Set value
                self.colors[node] = [value_try]
                self.values[value_try].append(node)
                #Check if all variables are assigned
                if not self.variables_left:
                    return True

                #Try value
                success = self.Search(new_variable, variable_heuristic, value_heuristic, discrepancies_left - 1 if used_heuristic else discrepancies_left, depth + 1, depth_limit)
                if success:
                    return True
                
                backtracks += 1
                #Reset color possible values for node and neighbors
                self.reset_color_possibilities(color_changes)
                self.values[value_try].remove(node)
                possible_values.remove(value_try)
                used_heuristic = True
                if value_pruning and not self.values[value_try]:
                    for value in self.values:
                        if not self.values[value] and value in possible_values:
                            possible_values.remove(value)

            else:
                break
        self.variables_left.append(node)
        return False
   
    def LDS_Search(self, variable_heuristic, value_heuristic):
        number_variables = len(self.graph.neighbors)
        for discrepancies in range(number_variables):
            if self.Search(variable_heuristic(), variable_heuristic, value_heuristic, discrepancies, 1, math.inf, ilds=True) == True:
                return self.finish_search()
            self.reset_state()
        return False
    # Depth-bounded Discrepancy Search
    def DDS_Search(self, variable_heuristic, value_heuristic, discrepancy_bound=math.inf):
        #Depth level for DDS search. A depth level of n means the heuristic is used in all but the first n nodes.
        #Don't use the value the Heuristic likes the most on the nth depth in the tree, as this has been repeated in previous iterations
        dds_depth = 0
        number_variables = len(self.graph.neighbors)
        self.variable_heuristic = variable_heuristic
        self.value_heuristic = value_heuristic
        #Start with a depth of 0 and increase the depth for DDS Search
        while(dds_depth < number_variables):
            #Number of nodes that we will not use the heuristic for
            for number_of_non_heuristics in range(min(dds_depth, discrepancy_bound + 1)):
                if self.Search(variable_heuristic(), variable_heuristic, value_heuristic, number_of_non_heuristics, 1, dds_depth) == True:
                        return self.finish_search()
                self.reset_state()
            dds_depth += 1
        
        return False


    def Lookahead(self, node, variable_heuristic, value_heuristic, depth, depth_limit):
        #All variables have been assigned
        number_variables_left = len(self.variables_left)
        if number_variables_left == 0 or depth == depth_limit:
            return 1
        
        #Some variable has no possible assignment (In place backjumping)
        for variable in self.variables_left:
            if not self.colors[variable]:
                return 0
        
        #Get list of possible values to assign variable sorted by order to try
        possible_values = value_heuristic(node)
        number_possible_values = len(possible_values)
        #Find next variable to use
        self.variables_left.remove(node)
        new_variable = None
        if self.variables_left:
            new_variable = variable_heuristic()
        #Return values from search
        rvals = {}

        #Iterate through possible values
        while(possible_values):
            value_try = possible_values[0]
            #Change neighbors possible values
            self.adjust_color_possibilities(node, value_try)
            #Get changes to neighbors values
            color_changes = copy.deepcopy(self.recent_changes)
            #Set value
            self.colors[node] = [value_try]
            self.values[value_try].append(node)
            #Check if all variables are assigned
            if not self.variables_left:
                if depth != 0:
                    return 1
                else:
                    self.reset_color_possibilities(color_changes)
                    self.values[value_try].remove(node)
                    possible_values.remove(value_try)
                    self.variables_left.append(node)
                    return {value: value == value_try for value in value_heuristic(node)}

            #Try value
            rvals[value_try] = self.Lookahead(new_variable, variable_heuristic, value_heuristic, depth + 1, depth_limit)
            #Reset color possible values for node and neighbors
            self.reset_color_possibilities(color_changes)
            self.values[value_try].remove(node)
            possible_values.remove(value_try)

        self.variables_left.append(node)
        if (depth == 0):
            return rvals
        return sum([rvals[color] for color in rvals]) / number_possible_values
    #Returns next variable to use
    def DFS_Variable_Heuristic(self):
        return self.variables_left[0]

    # Chooses next variable to use by selecting the variable with the least possible colors that it can choose
    def Least_Color_Choices_Variable_Heuristic(self):
        if not self.variables_left:
            return self.removed_variables[0]
        variables = sorted(self.variables_left, key = lambda x: (len(self.colors[x]), -self.graph.degree[x]))
        return variables[0]

    #Chooses next variable to use by selecting the variable with the Most Neighbors (slower than Least Color Choices)
    def Most_Neighbors_Variable_Heuristic(self):
        variables = sorted(self.variables_left, key = lambda x : (-self.graph.degree[x], len(self.colors[x])))
        return variables[0]

    #Returns list of values to try
    def DFS_Value_Heuristic(self, node):
        return copy.deepcopy(self.colors[node])

    # Returns list of values that a node can try
    # List is sorted in increasing order by the number of constraints that assigning the value to the node will create
    def Induced_Conflicts_Value_Heuristic(self, node):
        return sorted(self.colors[node], key = lambda x: (self.cost(node, x), -len(self.values[x])))
    
    def Used_Values_Value_Heuristic(self, node):
        return sorted(self.colors[node], key = lambda x: len(self.values[x]), reverse=True)

    def Lookahead_n(self, node, n):
        available_colors = self.DFS_Value_Heuristic(node)
        rvals = self.Lookahead(node, self.Least_Color_Choices_Variable_Heuristic, self.DFS_Value_Heuristic, 0, n)
        values = [color for color in available_colors if rvals[color] > 0]
        if not values:
            return []
        return sorted(values, key = lambda x: (-rvals[x], self.cost(node, x), -len(self.values[x])))

    def Lookahead_Value_Heuristic(self, node):
        return self.Lookahead_n(node, 4)

    def Regular_DFS_Search(self):
        if self.Search(self.DFS_Variable_Heuristic(), self.DFS_Variable_Heuristic, self.DFS_Value_Heuristic, math.inf, 1, math.inf):
            return self.finish_search()
        return False

    def Heuristic_DFS_Search(self):
        if self.Search(self.Least_Color_Choices_Variable_Heuristic(), self.Least_Color_Choices_Variable_Heuristic, self.Lookahead_Value_Heuristic, math.inf, 1, math.inf, value_pruning=True):
            return self.finish_search()
        return False

    def Limited_Discrepancy_Search(self):
        return self.LDS_Search(self.Least_Color_Choices_Variable_Heuristic, self.Used_Values_Value_Heuristic)

    def Depth_Bounded_Discrepancy_Search(self):
        return self.DDS_Search(self.Least_Color_Choices_Variable_Heuristic, self.Induced_Conflicts_Value_Heuristic, discrepancy_bound=5)

#self.colors[node] is the color of the node
#self.nodes[color] is a list of nodes that the given color has
#self.max_color is the currently largest numbered color
class Local_Search_State():
    def __init__(self, graph, num_colors):
        self.graph = graph
        self.colors = {}
        self.nodes = {}
        self.num_colors = num_colors
        #Initialize Graph with each node being a different color
        for variable in range(1, self.graph.num_nodes + 1):
            self.colors[variable] = variable
            self.nodes[variable] = [variable]
            self.max_color = self.graph.num_nodes
    #Resets colors chosen 
    def reset_state(self):
        for variable in range(1, self.graph.num_nodes + 1):
            self.colors[variable] = variable
            self.nodes[variable] = [variable]
            self.max_color = self.graph.num_nodes

    def Local_Search(self, num_colors, iterations, variable_heuristic, value_heuristic):
        
        if num_colors >= self.max_color:
            return True
        resets = 0
        while True:
            for iteration_number in range(iterations):
                variable = variable_heuristic()
                value = value_heuristic(variable)
                if value == None:
                    continue

                old_value = self.colors[variable]
                self.nodes[old_value].remove(variable)
                self.colors[variable] = value
                self.nodes[value].append(variable)
                #Check if any nodes have the value that we removed
                if not self.nodes[old_value]:
                    #If it is the largest number color, reduce the color count by 1
                    if old_value == self.max_color:
                        del self.nodes[self.max_color]
                        self.max_color -= 1
                    else:
                        #If it is not the largest number color take the nodes that have the largest number color,
                        #and set them to the color with no nodes. Then reduce the color count by 1
                        for node in self.nodes[self.max_color]:
                            self.colors[node] = old_value
                            self.nodes[old_value].append(node)
                        del self.nodes[self.max_color]
                        self.max_color -= 1
                    #Have we reduced the colors in the graph to the number of colors we want?
                    if num_colors == self.max_color:
                        print('Resets: ', resets)
                        print('Iterations: ', iteration_number)
                        print()
                        return True

            resets += 1
            self.reset_state()

    def pick_variable(self):
        choice = random.randint(0, 1)
        num_nodes_with_color = {color: len(self.nodes[color]) for color in self.nodes}
        node_heuristic = [1 / num_nodes_with_color[self.colors[node]] for node in self.colors]
        if choice == 0:
            return random.choices([color for color in self.colors], weights = node_heuristic)[0]
        else:
            min_color = min([num_nodes_with_color[color] for color in num_nodes_with_color])
            for color in self.nodes:
                if len(self.nodes[color]) == min_color:
                    return self.nodes[color][0]

    def pick_neighbor_variable(self):
        choice = random.randint(0, 2)
        num_nodes_with_color = {color: len(self.nodes[color]) for color in self.nodes}
        if choice == 0:
            neighbor_colors = {node : len(set([self.colors[neighbor] for neighbor in self.graph.neighbors[node]])) for node in self.colors}
            max_length = max([neighbor_colors[node] for node in self.colors])
            for node in self.colors:
                if neighbor_colors[node] == max_length:
                    return node
            
        elif choice == 1:
            min_color = min([num_nodes_with_color[color] for color in num_nodes_with_color])
            for color in self.nodes:
                if len(self.nodes[color]) == min_color:
                    return self.nodes[color][0]
        else:
            node_heuristic = [1 / num_nodes_with_color[self.colors[node]] for node in self.colors]
            return random.choices([color for color in self.colors], weights = node_heuristic)[0]

    def pick_value(self, node):
        choice = random.randint(0,1)
        neighbor_colors = {self.colors[neighbor] for neighbor in self.graph.neighbors[node]}
        num_nodes_with_color = {color : len(self.nodes[color]) for color in self.nodes if not color in neighbor_colors}
        value_heuristic = [num_nodes_with_color[color] for color in num_nodes_with_color]
        if not value_heuristic:
            return None
        if choice != 0:
            return random.choices([color for color in num_nodes_with_color], weights = value_heuristic)[0]

        else:
            max_color = max([num_nodes_with_color[color] for color in num_nodes_with_color])
            for color in self.nodes:
                if len(self.nodes[color]) == max_color:
                    return color

    def Search(self):
        return self.Local_Search(self.num_colors, 100000, self.pick_neighbor_variable, self.pick_value)

def main():
    sys.setrecursionlimit(2000)
    global num_nodes_visited
    global backtracks
    num_nodes_visited = 0
    backtracks = 0
    state = None
    search = None

    options, args = getopt.getopt(sys.argv[1:], "", ["DFS", "Heuristic", "LDS", "DDS", "Local"])
    if len(args) != 2:
        print('Usage: ', sys.argv[0], ' [-options][graph_file][n]')
        sys.exit()
    
    graph = Graph(args[0])
    number_colors = int(args[1])

    for opt, _ in options:
        if opt == '--DFS':
            state = DFS_State(graph, number_colors)
            search = state.Regular_DFS_Search
        elif opt == '--Heuristic':
            state = DFS_State(graph, number_colors)
            search = state.Heuristic_DFS_Search
        elif opt == '--LDS':
            state = DFS_State(graph, number_colors)
            search = state.Limited_Discrepancy_Search
        elif opt == '--DDS':
            state = DFS_State(graph, number_colors)
            search = state.Depth_Bounded_Discrepancy_Search
        elif opt == '--Local':
            state = Local_Search_State(graph, number_colors)
            search = state.Search

    success = search()
    if not success:
        print('No such coloring found with ' + sys.argv[2] + ' colors')
    elif isinstance(state, Local_Search_State):
        for color in state.nodes:
            print('color', color, ':', end=' ')
            for value in state.nodes[color]:
                print(value, end=' ')
            print()
    elif isinstance(state, DFS_State):
        for color in state.values:
            print('color', color, ':', end=' ')
            for value in state.values[color]:
                print(value, end=' ')
            print()
        print()
        print('Variables Visited: ', num_nodes_visited)
        print('Number of Backtracks: ', backtracks)

if __name__ == '__main__':
    main()
