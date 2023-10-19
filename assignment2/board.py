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
import threading


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
                if color == BLACK:
                    self.black_captures += 2
                else:
                    self.white_captures += 2
        if(len(capturesList) > 0):
            capturesList.append(O)
            self.capture_stack.append(capturesList)
        
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
        #for stone in list:
            #if self.get_color(stone) == prev:
                #counter += 1
            #else:
                #counter = 1
                #prev = self.get_color(stone)
                #if()
            #if counter == 5 and prev != EMPTY:
                #return prev
        for i in range(0,len(list)):
            if self.get_color(list[i]) == prev:
                counter += 1
            else:
                if(len(list)-i < 5):
                    return EMPTY
                counter = 1
                prev = self.get_color(list[i])
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY
    
    def undo_move(self, point, capture):
        # undo captures and capture counts
        if capture:
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
            score = self.getHeuristicScore()
            
        return score
    
    def getHeuristicScore(self):
        score = 0
        opp = opponent(self.current_player)
        lines = self.rows + self.cols + self.diags
        for line in lines:
            for i in range(len(line) - 5):
                currentPlayerCount = 0
                opponentCount = 0
                # count the number of stones on each five-line
                for p in line[i:i + 5]:
                    if self.board[p] == self.current_player:
                        currentPlayerCount += 1
                    elif self.board[p] == opp:
                        opponentCount += 1
                # Is blocked
                if currentPlayerCount < 1 or opponentCount < 1:
                    score += 10 ** currentPlayerCount - 10 ** opponentCount
        return score


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
        direction = 'diagonal'

        if direction == 'diagonal':
            find_all = True
            while find_all:
                
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
                if self.get_color(nb_list[2]) == self.get_color(point):
                    NS += 1
                    nb_list[2] = self._neighbors(nb_list[2])[2]
                if self.get_color(nb_list[3]) == self.get_color(point):
                    NS += 1
                    nb_list[3] = self._neighbors(nb_list[3])[3]
                
                if self.get_color(point) not in [self.get_color(nb_list[2]), self.get_color(nb_list[3])]:
                    find_all = False
        
        return [NS, EW, NWSE, NESW] 
    '''
    def detect_n_in_row(self,n):
        
        #Checks for a group of n stones in the same direction on the board.
        w = []
        b = []
        for r in self.rows:
            rows = self.has_n_in_list(r,n)
            w += rows[0]
            b += rows[1]
        for c in self.cols:
            cols = self.has_n_in_list(c,n)
            w += cols[0]
            b += cols[1]
        for d in self.diags:
            diags = self.has_n_in_list(d,n)
            w += diags[0]
            b += diags[1]


        if self.current_player == BLACK:
            return b+w
        if self.current_player == WHITE:
            return w+b
        #return []
    '''
    def detect_n_in_row(self,n):
        
        #Checks for a group of n stones in the same direction on the board.
        b5 = []
        w5 = []
        b4 = []
        w4 = []
        b3 = []
        w3 = []
        for r in self.rows:
            rows = self.has_n_in_list(r)
            w5 += rows[0]
            w4 += rows[1]
            w3 += rows[2]
            b5 += rows[3]
            b4 += rows[4]
            b3 += rows[5]
        for c in self.cols:
            cols = self.has_n_in_list(c)
            w5 += cols[0]
            w4 += cols[1]
            w3 += cols[2]
            b5 += cols[3]
            b4 += cols[4]
            b3 += cols[5]
        for d in self.diags:
            diags = self.has_n_in_list(d)
            w5 += diags[0]
            w4 += diags[1]
            w3 += diags[2]
            b5 += diags[3]
            b4 += diags[4]
            b3 += diags[5]

        if self.current_player == BLACK:
            return b5+w5+b4+w4+b3+w3
        elif self.current_player == WHITE:
            return w5+b5+w4+b4+w3+b3
        #return []
    
    def has_n_in_list(self, list) -> GO_COLOR:
        """
        Checks if there are n stones in a row.
        Returns BLACK or WHITE if any n in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = self.board[list[0]]
        counter = 1
        empty = 0
        gap = 0
        b5 = []
        w5 = []
        b4 = []
        w4 = []
        b3 = []
        w3 = []
        for i in range(1,len(list)):
            color = self.board[list[i]]
            if color == prev:
                # Matching stone
                counter += 1
            elif(empty == 0 and color == EMPTY):
                # The stone is an empty space on the board.
                empty = i
                gap = counter

            elif(color == EMPTY and empty != i-1):
                    empty = i # reset empty and subtract the gap from the counter
                    counter = counter - gap
                    gap = counter
            else:
                if(len(list) - i <= 2):
                    return [w5,w4,w3,b5,b4,b3]
                gap = 0
                counter = 1
                empty = 0
                prev = color
            if(color != EMPTY and (i+1 >= len(list) or self.board[list[i+1]] != color)):
                if(counter == 2):
                    w3,b3 = self.set_space(w3,b3,empty,list,2,i,color)
                elif(counter == 3):
                    w4,b4 = self.set_space(w4,b4,empty,list,3,i,color)
                elif(counter == 4):
                    w5,b5 = self.set_space(w5,b5,empty,list,4,i,color)

        return [w5,w4,w3,b5,b4,b3]
    
    
    def set_space(self,w,b,empty,list,n,i,color):
        if(color == BLACK):
            if(empty > 0):
                b.append(list[empty])
                return [w,b]
            elif(n >= 4):
                if(i+1 < len(list) and self.board[list[i+1]] == EMPTY):
                    b.append(list[i+1])
                if(i-n >= 0 and self.board[list[i-n]] == EMPTY):
                    b.append(list[i-n])
                return [w,b]
            
            elif(self.current_player != BLACK):
                if(i+1 < len(list) and self.board[list[i+1]] == EMPTY and i-n >= 0 and self.board[list[i-n]] == EMPTY):
                    b.append(list[i+1])
                    b.append(list[i-n])
                return [w,b]
            else:
                if(i+2 < len(list) and self.board[list[i+1]] == EMPTY):
                    b.append(list[i+1])
                if(i-n-1 >= 0 and self.board[list[i-n]] == EMPTY):
                    b.append(list[i-n])
                return [w,b]
            '''else:
                if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                    b.append(list[i+1])
                if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                    b.append(list[i-n])'''
        elif(color == WHITE):
            if(empty > 0):
                w.append(list[empty])
                return [w,b]

            elif(n >= 4):
                if(i+1 < len(list) and self.board[list[i+1]] == EMPTY):
                    w.append(list[i+1])
                if(i-n >= 0 and self.board[list[i-n]] == EMPTY):
                    w.append(list[i-n]) 
                return [w,b]   
            
            elif(self.current_player != WHITE):
                if(i + 1 < len(list) and self.board[list[i+1]] == EMPTY and i-n >= 0 and self.board[list[i-n]] == EMPTY):
                    w.append(list[i+1])
                    w.append(list[i-n])
                return [w,b]

            else:
                if(i+2 < len(list) and self.board[list[i+1]] == EMPTY):
                    w.append(list[i+1])
                if(i-n-1 >= 0 and self.board[list[i-n]] == EMPTY):
                    w.append(list[i-n])
        return [w,b]
        '''else:
                if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                    w.append(list[i+1])
                if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                    w.append(list[i-n])'''
    

    '''def set_space(self,w,b,empty,list,n,i):
        if(self.get_color(list[i]) == BLACK):
            if(empty > 0):
                if(n == 4):
                    b.append(list[empty])
                else:
                    count = 0
                    for j in range(1,5-n):
                        if(i+j >= len(list) or self.get_color(list[j+i]) == WHITE):
                            break
                        count+=1    
                    for j in range(0,5-n-1):
                        if(i-n-j-1 < 0 or self.get_color(list[i-n-j-1]) == WHITE):
                            break
                        count += 1
                    if(count + n + 1 >= 5):
                        b.append(list[empty])
            else:
                if(n == 4):
                    if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                        b.append(list[i+1])
                    if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                        b.append(list[i-n])
                else:
                    count = 0
                    for j in range(1,5-n+1):
                        if(i+j >= len(list) or self.get_color(list[j+i]) == WHITE):
                            break
                        count+=1    
                    for j in range(0,5-n):
                        if(i-n-j < 0 or self.get_color(list[i-n-j]) == WHITE):
                            break
                        count += 1
                    if(count + n >= 5):
                        if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                            b.append(list[i+1])
                        if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                            b.append(list[i-n])
        elif(self.get_color(list[i]) == WHITE):
            if(empty > 0):
                if(n == 4):
                    b.append(list[empty])
                else:
                    count = 0
                    for j in range(1,5-n):
                        if(i+j >= len(list) or self.get_color(list[j+i]) == WHITE):
                            break
                        count+=1    
                    for j in range(0,5-n-1):
                        if(i-n-j-1 < 0 or self.get_color(list[i-n-j-1]) == WHITE):
                            break
                        count += 1
                    if(count + n + 1 >= 5):
                        b.append(list[empty])
            else:
                if(n == 4):
                    if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                        w.append(list[i+1])
                    if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                        w.append(list[i-n])
                else:
                    count = 0
                    for j in range(1,5-n+1):
                        if(i+j >= len(list) or self.get_color(list[j+i]) == WHITE):
                            break
                        count+=1    
                    for j in range(0,5-n):
                        if(i-n-j < 0 or self.get_color(list[i-n-j]) == WHITE):
                            break
                        count += 1
                    if(count + n >= 5):
                        if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                            w.append(list[i+1])
                        if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                            w.append(list[i-n])
        return [w,b]'''

    '''def has_n_in_list(self, list, n) -> GO_COLOR:
        """
        Checks if there are n stones in a row.
        Returns BLACK or WHITE if any n in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        empty = 0
        gap = 0
        b = []
        w = []
        for i in range(len(list)):
            if self.get_color(list[i]) == prev:
                # Matching stone
                counter += 1
            elif(empty == 0 and self.get_color(list[i]) == EMPTY):
                # The stone is an empty space on the board.
                empty = i
                gap = counter
            else:
                empty = 0
                gap = 0
                counter = 1
                prev = self.get_color(list[i])
            if(counter == n-1):
                if(self.get_color(list[i]) == BLACK):
                    if(empty > 0):
                        b.append(list[empty])
                        empty = 0 # reset empty and subtract the gap from the counter
                        counter = counter - gap
                    else:
                        if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                            b.append(list[i+1])
                        if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                            b.append(list[i-n])
                elif(self.get_color(list[i]) == WHITE):
                    if(empty > 0):
                        w.append(list[empty])
                        empty = 0 # reset empty and subtract the gap from the counter
                        counter = counter - gap
                    else:
                        if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                            w.append(list[i+1])
                        if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                            w.append(list[i-n])
            #if counter == n-1 and prev != EMPTY:
                # This is working for 3,4 in a row
                ##################################################
                # if (i+1-n) > 0:
                #     if self.get_color(list[i-n]) == EMPTY:
                #         if self.get_color(list[i]) == BLACK:
                #             b.append(list[i-n])
                #         elif self.get_color(list[i]) == WHITE:
                #             w.append(list[i-n])
                # if (i+1) < self.size:
                #     if self.get_color(list[i+1]) == EMPTY:
                #         if self.get_color(list[i]) == BLACK:
                #             b.append(list[i+1])
                #         elif self.get_color(list[i]) == WHITE:
                #             w.append(list[i+1])
                #######################################################33

                ## this is not behaving exactly as expected but a it sholud work if we can fix it
                ## it works for some cases like a1,a2,a4 gives a3 as the move
                print(i+2-n)
                if (i+2-n) > 0:
                    if self.get_color(list[i-n]) == EMPTY and self.get_color(i-n-1) == self.get_color(list[i]):
                        if self.get_color(list[i]) == BLACK:
                            b.insert(0,list[i-n])
                        elif self.get_color(list[i]) == WHITE:
                            w.insert(0,list[i-n])
                elif (i+1-n) > 0:
                    if self.get_color(list[i-n]) == EMPTY:
                        if self.get_color(list[i]) == BLACK:
                            b.append(list[i-n])
                        elif self.get_color(list[i]) == WHITE:
                            w.append(list[i-n])
                # check above
                if (i+2) < self.size:
                    if self.get_color(list[i+1]) == EMPTY and self.get_color(list[i+2]) == self.get_color(list[i]):
                        if self.get_color(list[i]) == BLACK:
                            b.insert(0,list[i+1])
                        elif self.get_color(list[i]) == WHITE:
                            w.insert(0,list[i+1])
                elif (i+1) < self.size:
                    if self.get_color(list[i+1]) == EMPTY:
                        if self.get_color(list[i]) == BLACK:
                            b.append(list[i+1])
                        elif self.get_color(list[i]) == WHITE:
                            w.append(list[i+1])
        return [w,b]'''
    
    '''
    def detect_n_in_row_dict(self,n):
        
        #Checks for a group of n stones in the same direction on the board.
        
        w,b = {},{}
        #w,b = [],[]
        for r in self.rows:
            rows = self.has_n_in_list(r,n,w,b)
            #w += rows[0]
            #b += rows[1]
            w = rows[0]
            b = rows[1]
        for c in self.cols:
            cols = self.has_n_in_list(c,n,w,b)
            #w += cols[0]
            #b += cols[1]
            w = rows[0]
            b = rows[1]
        for d in self.diags:
            diags = self.has_n_in_list(d,n,w,b)
            #w += diags[0]
            #b += diags[1]
            w = rows[0]
            b = rows[1]
        if self.current_player == BLACK:
            b = list(dict(sorted(b.items(), key = operator.itemgetter(1), reverse = True)).keys())
            b.reverse()
            w = list(dict(sorted(w.items(), key = operator.itemgetter(1), reverse = True)).keys())
            w.reverse()
            return list(dict.fromkeys(b + w))
            #return b+list((set(w)-set(b)))
        if self.current_player == WHITE:
            b = list(dict(sorted(b.items(), key = operator.itemgetter(1))).keys())
            b.reverse()
            w = list(dict(sorted(w.items(), key = operator.itemgetter(1))).keys())
            w.reverse()
            return list(dict.fromkeys(w + b))
            #return w+list((set(b)-set(w)))

    def has_n_in_list_dict(self, list, n, w, b) -> GO_COLOR:
        """
        Checks if there are n stones in a row.
        Returns BLACK or WHITE if any n in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        empty = 0
        gap = 0
        #b = []
        #w = []
        for i in range(len(list)):
            if self.get_color(list[i]) == prev:
                # Matching stone
                counter += 1
            elif(empty == 0 and self.get_color(list[i]) == EMPTY):
                # The stone is an empty space on the board.
                empty = i
                gap = counter
            else:
                empty = 0
                gap = 0
                counter = 1
                prev = self.get_color(list[i])
            if(counter == n-1):
                if(self.get_color(list[i]) == BLACK):
                    if(empty > 0):
                        #b.append(list[empty])
                        if(list[empty] in b):
                            b[list[empty]] += 1
                        else:
                            b[list[empty]] = 1
                        empty = 0 # reset empty and subtract the gap from the counter
                        counter = counter - gap
                    else:
                        if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                            #b.append(list[i+1])
                            if(list[i+1] in b):
                                b[list[i+1]] += 1
                            else:
                                b[list[i+1]] = 1
                        if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                            #b.append(list[i-n])
                            if(list[i-n] in b):
                                b[list[i-n]] += 1
                            else:
                                b[list[i-n]] = 1
                elif(self.get_color(list[i]) == WHITE):
                    if(empty > 0):
                        #w.append(list[empty])
                        if(list[empty] in w):
                            w[list[empty]] += 1
                        else:
                            w[list[empty]] = 1
                        empty = 0 # reset empty and subtract the gap from the counter
                        counter = counter - gap
                    else:
                        if(i+1 < len(list) and self.get_color(list[i+1]) == EMPTY):
                            #w.append(list[i+1])
                            if(list[i+1] in w):
                                w[list[i+1]] += 1
                            else:
                                w[list[i+1]] = 1
                        if(i-n >= 0 and self.get_color(list[i-n]) == EMPTY):
                            #w.append(list[i-n])
                            if(list[i-n] in w):
                                w[list[i-n]] += 1
                            else:
                                w[list[i-n]] = 1
            #if counter == n-1 and prev != EMPTY:
                # This is working for 3,4 in a row
                ##################################################
                # if (i+1-n) > 0:
                #     if self.get_color(list[i-n]) == EMPTY:
                #         if self.get_color(list[i]) == BLACK:
                #             b.append(list[i-n])
                #         elif self.get_color(list[i]) == WHITE:
                #             w.append(list[i-n])
                # if (i+1) < self.size:
                #     if self.get_color(list[i+1]) == EMPTY:
                #         if self.get_color(list[i]) == BLACK:
                #             b.append(list[i+1])
                #         elif self.get_color(list[i]) == WHITE:
                #             w.append(list[i+1])
                #######################################################33

                ## this is not behaving exactly as expected but a it sholud work if we can fix it
                ## it works for some cases like a1,a2,a4 gives a3 as the move
                print(i+2-n)
                if (i+2-n) > 0:
                    if self.get_color(list[i-n]) == EMPTY and self.get_color(i-n-1) == self.get_color(list[i]):
                        if self.get_color(list[i]) == BLACK:
                            b.insert(0,list[i-n])
                        elif self.get_color(list[i]) == WHITE:
                            w.insert(0,list[i-n])
                elif (i+1-n) > 0:
                    if self.get_color(list[i-n]) == EMPTY:
                        if self.get_color(list[i]) == BLACK:
                            b.append(list[i-n])
                        elif self.get_color(list[i]) == WHITE:
                            w.append(list[i-n])
                # check above
                if (i+2) < self.size:
                    if self.get_color(list[i+1]) == EMPTY and self.get_color(list[i+2]) == self.get_color(list[i]):
                        if self.get_color(list[i]) == BLACK:
                            b.insert(0,list[i+1])
                        elif self.get_color(list[i]) == WHITE:
                            w.insert(0,list[i+1])
                elif (i+1) < self.size:
                    if self.get_color(list[i+1]) == EMPTY:
                        if self.get_color(list[i]) == BLACK:
                            b.append(list[i+1])
                        elif self.get_color(list[i]) == WHITE:
                            w.append(list[i+1])
        return [w,b]
    '''

'''
DELETE LATER
'''

def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is transformed to (PASS,PASS)
    """
    if point == PASS:
        return (PASS, PASS)
    else:
        NS = boardsize + 1
        point = divmod(point, NS)
        return format_point(point)

def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1', or 'PASS'.
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if move[0] == PASS:
        return "PASS"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1] + str(row)

def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return (PASS, PASS)
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError("wrong coordinate")
    if not (col <= board_size and row <= board_size):
        raise ValueError("wrong coordinate")
    return coord_to_point(row, col,board_size)



def printMoves(list):
    print (" ".join([point_to_coord(word, 7) for word in list]))