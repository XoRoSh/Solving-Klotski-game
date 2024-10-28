import heapq
import time
import argparse
from collections import deque
import os
import copy
import sys
import collections

#====================================================================================

char_single = '2'
goal_board = None
height=None

class Piece:
    """ This represents a piece on the Hua Rong Dao puzzle.  """

    def __init__(self, is_2_by_2, is_single, coord_x, coord_y, orientation):
        """
        :param is_2_by_2: True if the piece is a 2x2 piece and False otherwise.
        :type is_2_by_2: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v') 
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_2_by_2 = is_2_by_2
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation
        if self.is_single:
            self.width = 1
        elif self.is_2_by_2:
            self.width = 2
        elif self.orientation == 'h':
            self.width = 2
        else:
            self.width = 1

        if self.is_single:
            self.height = 1
        elif self.is_2_by_2:
            self.height = 2
        elif self.orientation == 'h':
            self.height = 1
        else:
            self.height = 2

    def set_coords(self, coord_x, coord_y):
        """
        Move the piece to the new coordinates. 

        :param coord: The new coordinates after moving.
        :type coord: int
        """

        self.coord_x = coord_x
        self.coord_y = coord_y
    





    def manhattan_distance(self, coord_x, coord_y): 
        return abs(self.coord_x - coord_x) + abs(self.coord_y - coord_y)


    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_2_by_2, self.is_single, \
            self.coord_x, self.coord_y, self.orientation)

class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, height, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = height
        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

        self.blanks = []


    def can_move_right(self, piece):
        if piece.coord_x + piece.width >= self.width: 
            return False
        elif piece.is_single and self.grid[piece.coord_y][piece.coord_x + 1] != '.':
            return False
        elif piece.orientation == 'v' and (
            piece.coord_x + 1 < self.width and (self.grid[piece.coord_y][piece.coord_x + 1] != '.' or 
                                                self.grid[piece.coord_y + 1][piece.coord_x + 1] != '.')):
            return False
        elif piece.is_2_by_2 and (
            self.grid[piece.coord_y][piece.coord_x + 2] != '.' or 
            self.grid[piece.coord_y + 1][piece.coord_x + 2] != '.'):
            return False
        elif piece.orientation == 'h' and (
            self.grid[piece.coord_y][piece.coord_x + 2] != '.'):
            return False

        return True

    def can_move_left(self, piece):
        if piece.coord_x == 0:
            return False
        elif piece.is_single and self.grid[piece.coord_y][piece.coord_x - 1] != '.':
            return False
        elif piece.orientation == 'v' and (
            self.grid[piece.coord_y][piece.coord_x - 1] != '.' or self.grid[piece.coord_y + 1][piece.coord_x - 1] != '.'):
            return False
        elif piece.is_2_by_2 and (
            self.grid[piece.coord_y][piece.coord_x - 1] != '.' or 
            self.grid[piece.coord_y + 1][piece.coord_x - 1] != '.'):
            return False
        elif piece.orientation == 'h' and (
            self.grid[piece.coord_y][piece.coord_x - 1] != '.'):

            return False

        return True

    def can_move_up(self, piece):
        if piece.coord_y == 0:
            return False
        elif piece.is_single and self.grid[piece.coord_y - 1][piece.coord_x] != '.':
            return False
        elif piece.orientation == 'h' and (
            self.grid[piece.coord_y - 1][piece.coord_x] != '.' or self.grid[piece.coord_y - 1][piece.coord_x + 1] != '.'):
            return False
        elif piece.is_2_by_2 and (
            self.grid[piece.coord_y - 1][piece.coord_x] != '.' or 
            self.grid[piece.coord_y - 1][piece.coord_x + 1] != '.'):
            return False
        elif piece.orientation == 'v' and (
            self.grid[piece.coord_y - 1][piece.coord_x] != '.'):
            return False

        return True

    def can_move_down(self, piece):
        if piece.coord_y + piece.height >= self.height:
            return False
        elif piece.is_single and self.grid[piece.coord_y + 1][piece.coord_x] != '.':
            return False
        elif piece.orientation == 'h' and (
            self.grid[piece.coord_y + 1][piece.coord_x] != '.' or self.grid[piece.coord_y + 1][piece.coord_x + 1] != '.'):
            return False
        elif piece.is_2_by_2 and (
            self.grid[piece.coord_y + 2][piece.coord_x] != '.' or 
            self.grid[piece.coord_y + 2][piece.coord_x + 1] != '.'):
            return False
        elif piece.orientation == 'v' and (
            self.grid[piece.coord_y + 2][piece.coord_x] != '.'):
            return False

        return True

    # customized eq for object comparison.
    def __eq__(self, other):
        if isinstance(other, Board):
            if(grid_to_string(other) == grid_to_string(self)):
                return True
        return False


    def find_heuristic(self, goal_board):
        total_distance = 0
        for piece in self.pieces:
            goal_piece = None
            for goal in goal_board.pieces:
                goal_piece = goal
                break
            if goal_piece is not None:
                total_distance += piece.manhattan_distance(goal_piece.coord_x, goal_piece.coord_y)
            else:
                print("ERROR")
        return total_distance


    def __construct_grid(self):
        """
        called in __init__ to set up a 2-d grid based on the piece location information.

        """
        self.grid = []

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_2_by_2:
                self.grid[piece.coord_y][piece.coord_x] = '1'
                self.grid[piece.coord_y][piece.coord_x + 1] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = '1'
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'
    
    def create_board_children(self):
        children = []
        for piece in self.pieces:
            if self.can_move_right(piece): 
                new_pieces = copy.deepcopy(self.pieces)
                new_board = Board(self.height, new_pieces)
                piece_index = self.pieces.index(piece)
                new_board.pieces[piece_index].coord_x += 1
                new_board.__construct_grid()
                children.append(Board(self.height, new_board.pieces))
            if self.can_move_left(piece): 
                new_pieces = copy.deepcopy(self.pieces)
                new_board = Board(self.height, new_pieces)
                piece_index = self.pieces.index(piece)
                new_board.pieces[piece_index].coord_x -= 1
                new_board.__construct_grid()
                children.append(Board(self.height, new_board.pieces))
            if self.can_move_up(piece): 
                new_pieces = copy.deepcopy(self.pieces)
                new_board = Board(self.height, new_pieces)
                piece_index = self.pieces.index(piece)
                new_board.pieces[piece_index].coord_y -= 1
                new_board.__construct_grid()
                children.append(Board(self.height, new_board.pieces))
            if self.can_move_down(piece): 
                new_pieces = copy.deepcopy(self.pieces)
                new_board = Board(self.height, new_pieces)
                piece_index = self.pieces.index(piece)
                new_board.pieces[piece_index].coord_y += 1
                new_board.__construct_grid()
                children.append(Board(self.height, new_board.pieces))
        return children 





      

    def heuristic_function():
        for piece in self.pieces:
            pass

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()
        
        

class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """


    def __init__(self, board, hfn=float('inf'), f=0, depth=0, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param hfn: The heuristic function.
        :type hfn: Optional[Heuristic]
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.hfn = hfn
        self.f = f
        self.depth = depth
        self.parent = parent
        
    def __lt__(self, other):
            """Comparison method for the priority queue."""
            return self.f < other.f 

    def create_children(self): 

        children = list()
        for child_board in self.board.create_board_children():
            children.append(State(child_board, child_board.find_heuristic(goal_board),self.depth, self.depth + 1, self))
        return children
    
    def create_children_with_heuristic(self):
        children = list()
        for child_board in self.board.create_board_children():
            heuristic = child_board.find_heuristic(goal_board)
            g = self.depth
            children.append(State(child_board, heuristic, heuristic + g, g + 1, self))
        return children

    def is_goal(self): 
        return grid_to_string(self.board.grid) == grid_to_string(goal_board.grid)
        


def a_star(initial_state):
    """A* search algorithm to find the solution."""
    frontier = [] 
    heapq.heappush(frontier, (initial_state.f, initial_state))  
    pruning_states = set()  # Set to store visited states
    
    while frontier:
        current_f, current_state = heapq.heappop(frontier) 
        if current_state.is_goal():
            return current_state
        current_state_hash = hash(grid_to_string(current_state.board.grid))

        if current_state_hash in pruning_states:
            continue 
        pruning_states.add(current_state_hash)

        children = current_state.create_children_with_heuristic()
        for child in children:
            child_hash = hash(grid_to_string(child.board.grid))
            if child_hash not in pruning_states:
                heapq.heappush(frontier, (child.f, child)) 

    return None 



def dfs_solution(initial_state): 
    frontier = [initial_state] 
    pruning_states = set()

    while frontier: 
        current_state = frontier.pop()

        current_state_hash = hash(grid_to_string(current_state.board.grid))
        pruning_states.add(current_state_hash)

        if current_state.is_goal(): 
            return current_state 
        
        children = current_state.create_children()

        for state_n in children:
            state_n_hash = hash(grid_to_string(state_n.board.grid)) 
            if state_n_hash not in pruning_states:
                pruning_states.add(state_n_hash)
                frontier.append(state_n)
    return current_state; 






    



def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    global goal_board
    global height
    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    final_pieces = []
    final = False
    found_2by2 = False
    finalfound_2by2 = False
    height_ = 0

    for line in puzzle_file:
        height_ += 1
        if line == '\n':
            if not final:
                height_ = 0
                final = True
                line_index = 0
            continue
        if not final: #initial board
            for x, ch in enumerate(line):
                if ch == '^': # found vertical piece
                    pieces.append(Piece(False, False, x, line_index, 'v'))
                elif ch == '<': # found horizontal piece
                    pieces.append(Piece(False, False, x, line_index, 'h'))
                elif ch == char_single:
                    pieces.append(Piece(False, True, x, line_index, None))
                elif ch == '1':
                    if found_2by2 == False:
                        pieces.append(Piece(True, False, x, line_index, None))
                        found_2by2 = True
        else: #goal board
            for x, ch in enumerate(line):
                if ch == '^': # found vertical piece
                    final_pieces.append(Piece(False, False, x, line_index, 'v'))
                elif ch == '<': # found horizontal piece
                    final_pieces.append(Piece(False, False, x, line_index, 'h'))
                elif ch == char_single:
                    final_pieces.append(Piece(False, True, x, line_index, None))
                elif ch == '1':
                    if finalfound_2by2 == False:
                        final_pieces.append(Piece(True, False, x, line_index, None))
                        finalfound_2by2 = True
        line_index += 1
        
    puzzle_file.close()
    board = Board(height_, pieces)
    goal_board = Board(height_, final_pieces)
    return board, goal_board


def grid_to_string(grid):
    string = ""
    for i, line in enumerate(grid):
            for ch in line:
                string += ch
            string += "\n"
    return string 

def write_to_file(filename, string):
    f = open(filename, "a")
    string = string + "\n"
    f.write(string)
    f.close()

def recreate_solution(state): 
    current = state
    queue = deque()   



    while(current != None):
        text = grid_to_string(current.board.grid)
        queue.append(text)
        current = current.parent

    while queue: 
        write_to_file(args.outputfile, queue.pop()) 

    return None; 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()


    # read the board from the file
    board, goal_board = read_from_file(args.inputfile)
    height = board.height 
    initial_state = State(board, float('inf'), 0, 0, None)



    if(args.algo == 'dfs'):
        res = dfs_solution(initial_state)
        if res == None: 
            write_to_file(args.outputfile, "No solution")
        else:
            recreate_solution(res)
    elif(args.algo == 'astar'):
        res = a_star(initial_state)
        if res == None: 
            write_to_file(args.outputfile, "No solution")
        else:
            recreate_solution(res)
            
    else: 
        print("Wrong algorithm input")
    
