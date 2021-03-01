import random
import queue
import copy
import math
#This class contains a sliding tile puzzle
#self.dimensions is the width and height of the puzzle
#self.empty is the blank square
#self.puzzle is a list of the pieces in the puzzle
#For instance [0 2 None 1] would represent the puzzle:
#[0 2]
#[  1]
class Puzzle():

    def __init__(self, dimensions, puzzle=None, empty_square=None):
        self.dimensions = dimensions
        self.empty = empty_square
        self.puzzle = puzzle

    #Operators to allow us to put the puzzle in a set or queue.
    def __hash__(self):
        return hash(tuple(self.puzzle))

    def __lt__(self, obj):
        return hash(self) < hash(obj)

    def __eq__(self, obj):
        return self.puzzle == obj.puzzle

    #Creates random puzzle and checks to make sure it is solvable
    def create_random_puzzle(self):
        self.puzzle = [0] * self.dimensions * self.dimensions
        tile_pieces = [i for i in range(self.dimensions * self.dimensions - 1)]
        shuffled_pieces = random.sample(tile_pieces, k = len(tile_pieces))
        blank_tile = random.sample(list(range(self.dimensions * self.dimensions)), k = 1)[0]
        self.empty = blank_tile
        self.puzzle[blank_tile] = None
        indices = [i for i in range(self.dimensions * self.dimensions) if i != blank_tile]
        indices_index = 0
        while(indices_index < self.dimensions * self.dimensions - 1):
            self.puzzle[indices[indices_index]] = shuffled_pieces[indices_index]
            indices_index += 1

        #Only construct solvable puzzles!!!
        num_inversions = 0
        values = copy.deepcopy(self.puzzle)
        values.remove(None)
        for index in range(self.dimensions * self.dimensions - 2):
            for index_2 in range(index + 1, self.dimensions * self.dimensions - 1):
                if values[index] > values[index_2]:
                    num_inversions += 1
        if self.dimensions % 2 == 0:
            if (self.empty // self.dimensions) % 2 == 0 and num_inversions % 2 == 0:
                self.create_random_puzzle()
            if (self.empty // self.dimensions) % 2 == 1 and num_inversions % 2 == 1:
                self.create_random_puzzle()
        else:
            if num_inversions % 2 == 1:
                self.create_random_puzzle()


    #Writes puzzle to file
    def save_puzzle(self, file):
        with open(file, 'w') as f:
            for index in range(self.dimensions * self.dimensions):
                f.write(str(index) + ' ' + str(self.puzzle[index]) + '\n')

    #Loads puzzle from file
    def load_puzzle(self, file):
        self.puzzle = [0] * self.dimensions * self.dimensions
        with open(file, 'r') as f:
            readline = None
            while (readline != ''):
                readline = f.readline()
                if readline  != '':
                    readline = readline[:-1].split()
                    i, val = readline
                    if val == 'None':
                        self.puzzle[int(i)] = None
                        self.empty = int(i)
                    else:
                        self.puzzle[int(i)] = int(val)

    #Checks to see if the puzzle is solved
    def check_solved(self):
        for index in range(self.dimensions * self.dimensions - 1):
            if index != self.puzzle[index]:
                return False
        return True

    #Checks to see if a pattern (subset of the puzzle) is solved
    def pattern_solved(self, pattern):
        for index in pattern:
            if index != self.puzzle[index]:
                return False
        return True

    #The following up_move, down_move, left_move, right move changes the puzzle according to the proper move
    def up_move(self):
        i = self.empty // self.dimensions
        j = self.empty % self.dimensions
        if i < (self.dimensions - 1):
            self.puzzle[self.empty] = self.puzzle[(i+1) * self.dimensions + j]
            self.puzzle[(i+1) * self.dimensions + j] = None
            self.empty = (i+1) * self.dimensions + j

    def down_move(self):
        i = self.empty // self.dimensions
        j = self.empty % self.dimensions
        if i > 0:
            self.puzzle[self.empty] = self.puzzle[(i-1)* self.dimensions + j]
            self.puzzle[(i-1) * self.dimensions + j] = None
            self.empty = (i-1) * self.dimensions + j

    def left_move(self):
        i = self.empty // self.dimensions
        j = self.empty % self.dimensions
        if j < (self.dimensions - 1):
            self.puzzle[self.empty] = self.puzzle[i * self.dimensions + j+1]
            self.puzzle[i * self.dimensions + j+1] = None
            self.empty = i * self.dimensions + j+1

    def right_move(self):
        i = self.empty // self.dimensions
        j = self.empty % self.dimensions
        if j > 0:
            self.puzzle[self.empty] = self.puzzle[i * self.dimensions + j-1]
            self.puzzle[i * self.dimensions +  j-1] = None
            self.empty = i * self.dimensions + j-1

#class for Pattern Database
#self.disjoint patterns is a list of lists whose union is all the pieces of the puzzle
#self.database is the pattern database
#self.database[pattern number][:] holds all the minimum number of moves it takes to achieve a pattern
class Pattern_Database():
    def __init__(self, puzzle, disjoint_patterns):
        self.puzzle = puzzle
        self.disjoint_patterns = disjoint_patterns
        self.database = dict()
        #This creates "too large" of a database, but makes the math easier
        for index in range(len(disjoint_patterns)):
            self.database[index] = [math.inf] * sum([((puzzle.dimensions * puzzle.dimensions) ** (i+1)) for i in range(len(disjoint_patterns[index]) + 1)])
        
    #Give a puzzle state calculate what the index is in the database corresponding to the puzzle
    def index_into_database(self, state, pattern_number):
        index_sum = 0
        pattern = self.disjoint_patterns[pattern_number]
        for index in range(len(pattern)):
            number = pattern[index]
            position = state.puzzle.index(number)
            index_sum += position * ((state.dimensions * state.dimensions) ** index)
        index_sum += state.puzzle.index(None) * ((state.dimensions * state.dimensions) ** len(pattern))
        return index_sum

    #Save database to a file so we can use it later
    def save_patterns(self, file):
        with open(file, 'w') as f:
            for index in range(len(self.disjoint_patterns)):
                for val in self.disjoint_patterns[index]:
                    f.write(str(val) + ' ')
                f.write('\n')
                for val in self.database[index]:
                    f.write(str(val) + '\n')

    #Load database from a file (Much quicker than recalculating the values)
    def load_patterns_from_file(self, file):
        with open(file) as f:
            for index in range(len(self.disjoint_patterns)):
                f.readline()
                for value in range(len(self.database[index])):
                    val = f.readline()[:-1]
                    if val == 'inf':
                        self.database[index][value] = math.inf
                    else:
                        self.database[index][value] = int(val)

    #Returns neighbors to a puzzle along with whether the value we moved to get to the neighbor is in our pattern
    @staticmethod
    def neighbors(node, pattern):
        neighbors_list = []
        i = node.empty // node.dimensions
        j = node.empty % node.dimensions
        if j > 0:
            right_move_node = copy.deepcopy(node)
            right_move_node.right_move()
            moved_val = node.puzzle[i * node.dimensions + j - 1]
            neighbors_list.append((right_move_node, 1 if moved_val in pattern else 0))

        if j < (node.dimensions - 1):
            left_move_node = copy.deepcopy(node)
            left_move_node.left_move()
            moved_val = node.puzzle[i * node.dimensions + j + 1]
            neighbors_list.append((left_move_node, 1 if moved_val in pattern else 0))

        if i > 0:
            down_move_node = copy.deepcopy(node)
            down_move_node.down_move()
            moved_val = node.puzzle[(i-1) * node.dimensions + j]
            neighbors_list.append((down_move_node, 1 if moved_val in pattern else 0))

        if i < (node.dimensions - 1):
            up_move_node = copy.deepcopy(node)
            up_move_node.up_move()
            moved_val = node.puzzle[(i+1) * node.dimensions + j]
            neighbors_list.append((up_move_node, 1 if moved_val in pattern else 0))

        return neighbors_list

    #This function calculates how many moves have to be made to reach the goal state
    #Only moves that involve pieces in the pattern are counted
    #Only pieces in the pattern need to be in the correct state
    #Starts at the goal state and uses retrograde analysis to reach all the other states
    #States where empty state and pattern pieces are in the same position are considered to be the same 
    def load_pattern(self, pattern_number):
        pattern = self.disjoint_patterns[pattern_number]
        initial_state = [-1 if index not in pattern else index for index in range(self.puzzle.dimensions * self.puzzle.dimensions)]
        initial_state[-1] = None
        initial_puzzle = Puzzle(self.puzzle.dimensions, initial_state, self.puzzle.dimensions * self.puzzle.dimensions - 1)
        priority_queue = queue.PriorityQueue()
        priority_queue.put((0, initial_puzzle))
        stop_list = set()
        while not priority_queue.empty():
            cost, state = priority_queue.get()
            database_index = self.index_into_database(state, pattern_number)
            if cost < self.database[pattern_number][database_index]:
                self.database[pattern_number][database_index] = cost

            for neighbor, additional_cost in self.neighbors(state, pattern):
                neighbor_database_index = self.index_into_database(neighbor, pattern_number)
                if (cost + additional_cost) < self.database[pattern_number][neighbor_database_index]:
                    priority_queue.put((cost + additional_cost, neighbor))

            stop_list.add(state)


#class for State Space Search
#When done self.solve_state contains a string of the moves to bring the puzzle to the goal state 
class State_Search():
    def __init__(self, puzzle, patterns, file=None):
        self.puzzle = puzzle
        self.solve_state = None
        if file == None:
            self.database = Pattern_Database(puzzle, patterns)
            for index in range(len(patterns)):
                print('Loading pattern for ', patterns[index])
                self.database.load_pattern(index)
            print('Finished loading patterns')
        else:
            self.database = Pattern_Database(puzzle, patterns)
            self.database.load_patterns_from_file(file)

    #Returns neighbors of state, also gives information to which move was made to reach neighbor
    @staticmethod
    def neighbors(node):
        neighbors_list = []
        i = node.empty // node.dimensions
        j = node.empty % node.dimensions
        if j > 0:
            right_move_node = copy.deepcopy(node)
            right_move_node.right_move()
            neighbors_list.append((right_move_node, 'r'))

        if j < (node.dimensions - 1):
            left_move_node = copy.deepcopy(node)
            left_move_node.left_move()
            neighbors_list.append((left_move_node, 'l'))

        if i > 0:
            down_move_node = copy.deepcopy(node)
            down_move_node.down_move()
            neighbors_list.append((down_move_node, 'd'))

        if i < (node.dimensions - 1):
            up_move_node = copy.deepcopy(node)
            up_move_node.up_move()
            neighbors_list.append((up_move_node, 'u'))

        return neighbors_list

    #Heuristic for sum of manhattan distances
    @staticmethod
    def manhattan_distance(state):
        return sum([abs((state.puzzle[index] // state.dimensions) - (index // state.dimensions)) + abs((state.puzzle[index] % state.dimensions) - (index % state.dimensions)) for index in range(state.dimensions * state.dimensions) if index != state.empty])

    #Heuristic for summing up number of moves needed to take for each pattern
    def disjoint_pattern(self, state):
        pattern_sum = 0
        for pattern_number in range(len(self.database.disjoint_patterns)):
            database_index = self.database.index_into_database(state, pattern_number)
            pattern_sum += self.database.database[pattern_number][database_index]
        return pattern_sum

    #A* search to find solution state (bound is only used when called by IDA*)
    def A_star_search(self, bound = math.inf):
        stop_list = set()

        priority_queue = queue.PriorityQueue()
        priority_queue.put((self.disjoint_pattern(self.puzzle), (self.puzzle, 0, '')))
        while not priority_queue.empty():
            (_, (node, distance, path)) = priority_queue.get()
            if node in stop_list:
                continue
            if (node.check_solved()):
                self.solve_state = path
                return True

            for neighbor, direction in self.neighbors(node):
                if not neighbor in stop_list:
                    astar_distance = distance + self.disjoint_pattern(neighbor) + 1
                    if astar_distance <= bound:
                        priority_queue.put((astar_distance, (neighbor, distance + 1, path + direction)))
            stop_list.add(node)

        return False

    #IDA* search to find solution state
    def IDA_star_search(self):
        depth = 0
        found_solution = False
        while(not found_solution):
            print('Depth level: ', depth)
            found_solution = self.A_star_search(depth)
            depth += 1

            



def main():
    puzzle = Puzzle(4)
    puzzle.create_random_puzzle()
    print(puzzle.puzzle)

    #May have to use without file if you don't have the database file saved
    search = State_Search(puzzle, [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14]], '4-4-3-pattern.txt')
    search.IDA_star_search()
    print(search.solve_state, len(search.solve_state))
if __name__ == '__main__':
    main()