"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
import operator
from typing import List, Tuple

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT,
)


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0
        self.capture_stack = []

    def add_two_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            self.black_captures += 2
        elif color == WHITE:
            self.white_captures += 2
    def get_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            return self.black_captures
        elif color == WHITE:
            return self.white_captures
    
    def calculate_rows_cols_diags(self) -> None:
        if self.size < 5:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)
            
            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)
        
        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 5:
                self.diags.append(diag_NE)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >=5:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 5) + 1) * 2

    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.ko_recapture: GO_POINT = NO_POINT
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def _is_legal_check_simple_cases(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check the simple cases of illegal moves.
        Some "really bad" arguments will just trigger an assertion.
        If this function returns False: move is definitely illegal
        If this function returns True: still need to check more
        complicated cases such as suicide.
        """
        assert is_black_white(color)
        if point == PASS:
            return True
        # Could just return False for out-of-bounds, 
        # but it is better to know if this is called with an illegal point
        assert self.pt(1, 1) <= point <= self.pt(self.size, self.size)
        assert is_black_white_empty(self.board[point])
        if self.board[point] != EMPTY:
            return False
        if point == self.ko_recapture:
            return False
        return True

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        board_copy: GoBoard = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def end_of_game(self) -> bool:
        if self.get_empty_points().size == 0 or GO_COLOR(self.detect_five_in_a_row()) != EMPTY or self.black_captures >= 10 or self.white_captures >= 10:
            return True
        return False
    
    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def is_eye(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block: np.ndarray) -> bool:
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone: GO_POINT) -> np.ndarray:
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color: GO_COLOR = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point: GO_POINT) -> np.ndarray:
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=np.bool_)
        pointstack = [point]
        color: GO_COLOR = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point: GO_POINT) -> GO_POINT:
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns NO_POINT otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture: GO_POINT = NO_POINT
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture
    
    def play_move(self, point: GO_POINT, color: GO_COLOR) -> [bool, bool]:
        """
        Tries to play a move of color on the point.
        Returns whether or not the point was empty.
        """
        capture = False
        if self.board[point] != EMPTY:
            return False, capture
        self.board[point] = color
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        O = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS+1, -(self.NS+1), self.NS-1, -self.NS+1]
        capturesList = []
        for offset in offsets:
            if self.board[point+offset] == O and self.board[point+(offset*2)] == O and self.board[point+(offset*3)] == color:
                self.board[point+offset] = EMPTY
                self.board[point+(offset*2)] = EMPTY
                capture = True
                capturesList.append(point+offset)
                capturesList.append(point+(offset*2))
                # print(self.capture_stack)
                if color == BLACK:
                    self.black_captures += 2
                else:
                    self.white_captures += 2
        if(len(capturesList) > 0):
            capturesList.append(O)
            self.capture_stack.append(capturesList)
        # print(True, capture)
        #print(self.check_neighbours(point))
        self.detect_n_in_a_row(4)
        
        return True, capture
    
    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves

    def detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:        
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, list) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY
    
    def undo_move(self, point, capture):
        # undo captures and capture counts
        if capture:
            # print(self.capture_stack, ()"undo", capture)
            cap = self.capture_stack.pop(len(self.capture_stack)-1)
            color = cap[len(cap)-1]
            for i in range(len(cap)-1) :
                self.board[cap[i]] = GO_COLOR(color)
                if color == WHITE:
                    self.black_captures = self.black_captures - 1
                elif color == BLACK:
                    self.white_captures = self.white_captures - 1
        self.board[point] = EMPTY
        self.current_player = opponent(self.current_player)
        
    def legal_moves(self):
        moves = self.get_empty_points()
        return moves

    def staticallyEvaluateForToPlay(self) :
        captures = 0
        opp_captures = 0

        if(self.current_player == BLACK):
            captures = self.black_captures
            opp_captures = self.white_captures
        else:
            captures = self.white_captures
            opp_captures = self.black_captures

        if self.detect_five_in_a_row() == self.current_player or captures == 10:
            score = 100000000000

        elif self.detect_five_in_a_row() == opponent(self.current_player) or opp_captures == 10:
            score = -100000000000
        elif self.detect_five_in_a_row() == EMPTY:
            score = 0
            
        return score

    
    def generate_score(self, move):
        '''
        Need to generate scores to prioritize moves over one another in Alphabeta search.
        ORDER:
        1. A win, either 5 in a row or captured score == 10. POINTS: 5
        2. 4 in a row. POINTS: 4
        3. 3 in a row. POINTS: 3
        4. 2 in a row. POINTS: 2

        Other cases to consider:
        - Blocking an apponents 5 in a row should be more than a 4 in a row but less than a win. (maybe 4.5 points?)
        - A capture could be worth the same as 3 in a row? (Can change this in the future.)

        '''
        best_score = 0

        total_rows_list = self.in_row_count(self.rows)

        if self.detect_five_in_a_row():
            best_score = 5

    def check_neighbours(self, point):
        '''
        Function to count the number of matching neighbors in all directions.

        nb_list = [west, east, south, north, southwest, southeast, northwest, northeast]
        PAIRS:
            west+east -> 0,1
            south+north -> 2,3
            southwest+northeast -> 4,7
            southeast+northwest ->5,6

        '''
        nb_list = self._neighbors(point) + self._diag_neighbors(point)
        NS = 1
        EW = 1
        NWSE = 1
        NESW = 1
        #print(self.get_color(point))
        #print(point)
        direction = 'diagonal'
        # for nb in nb_list:
        #     print(nb)
        #     print(self.get_color(nb))

        if direction == 'diagonal':
            find_all = True
            while find_all:
                print("finding diagonal")
                
                if self.get_color(nb_list[4]) == self.get_color(point):
                    NESW += 1
                    nb_list[4] = self._diag_neighbors(nb_list[4])[0]
                    
                if self.get_color(nb_list[7]) == self.get_color(point):
                    NESW += 1
                    nb_list[7] = self._diag_neighbors(nb_list[7])[3]

                if self.get_color(nb_list[5]) == self.get_color(point):
                    NWSE += 1
                    nb_list[5] = self._diag_neighbors(nb_list[5])[1]

                if self.get_color(nb_list[6]) == self.get_color(point):
                    NWSE += 1
                    nb_list[6] = self._diag_neighbors(nb_list[6])[2]
                
                if self.get_color(point) not in [self.get_color(nb_list[4]), self.get_color(nb_list[5]), self.get_color(nb_list[6]), self.get_color(nb_list[7])]:
                    direction = 'horizontal'
                    find_all = False
                    
            
        if direction == 'horizontal':
            find_all = True
            while find_all:
                print("finding horizontal")
                if self.get_color(nb_list[0]) == self.get_color(point):
                    EW += 1
                    nb_list[0] = self._neighbors(nb_list[0])[0]
                if self.get_color(nb_list[7]) == self.get_color(point):
                    EW += 1
                    nb_list[1] = self._neighbors(nb_list[1])[1]
                
                if self.get_color(point) not in [self.get_color(nb_list[0]), self.get_color(nb_list[1])]:
                    direction = 'vertical'
                    find_all = False
            
        if direction == 'vertical':
            find_all = True
            while find_all:
                print("finding vertical")
                if self.get_color(nb_list[2]) == self.get_color(point):
                    NS += 1
                    nb_list[2] = self._neighbors(nb_list[2])[2]
                if self.get_color(nb_list[3]) == self.get_color(point):
                    NS += 1
                    nb_list[3] = self._neighbors(nb_list[3])[3]
                
                if self.get_color(point) not in [self.get_color(nb_list[2]), self.get_color(nb_list[3])]:
                    find_all = False
        
        return [NS, EW, NWSE, NESW] 
            
    
    

    def detect_n_in_a_row(self, n) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        #print("Entered detect_n_in_a_row")
        all_n = {1: None, 2: None, 3: None}
        counter = 0
        bd = {}
        wd = {}
        w2 = {}
        b2 = {}
        w3 = {}
        b3 = {}
        for r in self.rows:
            result, b, w, w_2, b_2, w_3, b_3 = self.has_n_in_list(r, n)
            #if result != EMPTY:
            if all_n[1] == None and result != EMPTY:
                all_n[1] = result
            elif result != EMPTY:
                all_n[1].append(result)
            
            for val in w:
                if val in wd:
                    wd[val] += 1
                else:
                    wd[val] = 1
            for val in b:
                if val in bd:
                    bd[val] += 1
                else:
                    bd[val] = 1
            for val in w_2:
                if val in w2:
                    w2[val] += 1
                else:
                    w2[val] = 1
            for val in b_2:
                if val in b2:
                    b2[val] += 1
                else:
                    b2[val] = 1
            for val in w_3:
                if val in w3:
                    w3[val] += 1
                else:
                    w3[val] = 1
            for val in b_3:
                if val in b3:
                    b3[val] += 1
                else:
                    b3[val] = 1
                # white_moves += list(set(w)-set(white_moves))
                # black_moves += list(set(b)-set(black_moves))
                counter += 1
                #return result
        for c in self.cols:
            result, b, w, w_2, b_2, w_3, b_3 = self.has_n_in_list(c, n)
            #if result != EMPTY:
            if all_n[2] == None and result != EMPTY:
                all_n[2] = result
            elif result != EMPTY:
                all_n[2].append(result)
            counter += 1
            for val in w:
                if val in wd:
                    wd[val] += 1
                else:
                    wd[val] = 1
            for val in b:
                if val in bd:
                    bd[val] += 1
                else:
                    bd[val] = 1
            for val in w_2:
                if val in w2:
                    w2[val] += 1
                else:
                    w2[val] = 1
            for val in b_2:
                if val in b2:
                    b2[val] += 1
                else:
                    b2[val] = 1
            for val in w_3:
                if val in w3:
                    w3[val] += 1
                else:
                    w3[val] = 1
            for val in b_3:
                if val in b3:
                    b3[val] += 1
                else:
                    b3[val] = 1
                # white_moves += list(set(w)-set(white_moves))
                # black_moves += list(set(b)-set(black_moves))
                #return result
        for d in self.diags:
            result, b, w, w_2, b_2, w_3, b_3 = self.has_n_in_list(d, n)
            #if result != EMPTY:
            if all_n[3] == None and result != EMPTY:
                all_n[3] = result
            elif result != EMPTY:
                all_n[3].append(result)
            counter += 1
            for val in w:
                if val in wd:
                    wd[val] += 1
                else:
                    wd[val] = 1
            for val in b:
                if val in bd:
                    bd[val] += 1
                else:
                    bd[val] = 1
            for val in w_2:
                if val in w2:
                    w2[val] += 1
                else:
                    w2[val] = 1
            for val in b_2:
                if val in b2:
                    b2[val] += 1
                else:
                    b2[val] = 1
            for val in w_3:
                if val in w3:
                    w3[val] += 1
                else:
                    w3[val] = 1
            for val in b_3:
                if val in b3:
                    b3[val] += 1
                else:
                    b3[val] = 1
                # white_moves += list(set(w)-set(white_moves))
                # black_moves += list(set(b)-set(black_moves))
                #return result

        

        print(w2,"\n", b2,"\n", w3,"\n", b3)
        return bd, wd
        #print("WHITE:", wd)
        #print("BLACK:", bd)
        #print(all_n[1], all_n[2], all_n[3])
        #print("Done detect_n_in_a_row")
        #return EMPTY

    def has_n_in_list(self, list, n) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        set = []
        all = []
        white_moves = []
        black_moves = []

        white_2 = []
        black_2 = []
        white_3 = []
        black_3 = []
        white_4 = []
        black_4 = []
        
        prev_stone = None
        # empties = []
        #print("enter")
        # if n == 1:
        #     print("enter")
        #     for stone in list:
        #         print("enter")
        #         s_color = self.get_color(stone)
        #         print("got color", s_color)
                
        #         if s_color != self.get_color(prev_stone) and s_color != 0:
                    
        #             set.append(stone)
        #         if s_color == 0 and prev_stone != None and prev_stone in set:
        #             print('there is an empty space by this stone')
        #         else:
        #             set = []
        #             counter = 1
        #             prev = self.get_color(stone)
        #             prev_stone = stone

        #else:
        for stone in list:
            

            if prev_stone == None:
                prev_stone = stone

            if self.get_color(stone) == 0 :
                if prev_stone in set:
                    if self.get_color(set[0]) == 1:
                        black_moves.append(stone)
                    elif self.get_color(set[0]) == 2:
                        white_moves.append(stone)
            if self.get_color(stone) == prev and self.get_color(stone) != 0:
                if set.count(prev) == 0:
                    set.append(prev_stone)
                    
                    
                set.append(stone)
                counter += 1
            else:
                if counter == 2:
                    potential_e = int(list.index(set[-1])-2)
                    if self.get_color(set[0]) == 2:
                        if self.get_color(stone) == 0:
                            white_2.append(stone)
                        if potential_e >= 0:
                            if self.get_color(list[potential_e]) == 0:
                                white_2.append(list[potential_e])
                    else:
                        if self.get_color(stone) == 0:
                            black_2.append(stone)
                        if potential_e >= 0:
                            if self.get_color(list[potential_e]) == 0:
                                black_2.append(list[potential_e])
                if counter == 3:
                    potential_e = int(list.index(set[-1])-3)
                    if self.get_color(set[0]) == 2:
                        if self.get_color(stone) == 0:
                            white_3.append(stone)
                        if potential_e >= 0:
                            if self.get_color(list[potential_e]) == 0:
                                white_3.append(list[potential_e])
                    else:
                        if self.get_color(stone) == 0:
                            black_3.append(stone)
                        if potential_e >= 0:
                            if self.get_color(list[potential_e]) == 0:
                                black_3.append(list[potential_e])
            
                set = []
                counter = 1
                prev = self.get_color(stone)
                prev_stone = stone

            if counter == n and prev != EMPTY and set not in all:
                potential_empty = int(list.index(set[n-1]) - n)
                if potential_empty >= 0:
                    prev_empty_val = list[potential_empty]
                    if self.get_color(prev_empty_val) == 0:
                        if self.get_color(set[0]) == 1:
                            black_moves.append(prev_empty_val)
                        elif self.get_color(set[0]) == 2:
                            white_moves.append(prev_empty_val)
                
                all.append(set)
        if len(white_2) > 0 or len(black_2) > 0:
            print("2, w:", white_2, "b:", black_2)
            print("3, w:", white_3, "b:", black_3)
            print("4, w:", white_4, "b:", black_4)
        
        if len(all) != 0:
            return all, black_moves, white_moves, white_2, black_2, white_3, black_3
        else:
            return EMPTY, black_moves, white_moves, white_2, black_2, white_3, black_3
    


'''
DELETE LATER
'''

# def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
#     """
#     Transform point given as board array index 
#     to (row, col) coordinate representation.
#     Special case: PASS is transformed to (PASS,PASS)
#     """
#     if point == PASS:
#         return (PASS, PASS)
#     else:
#         NS = boardsize + 1
#         point = divmod(point, NS)
#         return format_point(point)

# def format_point(move: Tuple[int, int]) -> str:
#     """
#     Return move coordinates as a string such as 'A1', or 'PASS'.
#     """
#     assert MAXSIZE <= 25
#     column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
#     if move[0] == PASS:
#         return "PASS"
#     row, col = move
#     if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
#         raise ValueError
#     return column_letters[col - 1] + str(row)

# def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
#     """
#     Convert a string point_str representing a point, as specified by GTP,
#     to a pair of coordinates (row, col) in range 1 .. board_size.
#     Raises ValueError if point_str is invalid
#     """
#     if not 2 <= board_size <= MAXSIZE:
#         raise ValueError("board_size out of range")
#     s = point_str.lower()
#     if s == "pass":
#         return (PASS, PASS)
#     try:
#         col_c = s[0]
#         if (not "a" <= col_c <= "z") or col_c == "i":
#             raise ValueError
#         col = ord(col_c) - ord("a")
#         if col_c < "i":
#             col += 1
#         row = int(s[1:])
#         if row < 1:
#             raise ValueError
#     except (IndexError, ValueError):
#         raise ValueError("wrong coordinate")
#     if not (col <= board_size and row <= board_size):
#         raise ValueError("wrong coordinate")
#     return coord_to_point(row, col,board_size)

