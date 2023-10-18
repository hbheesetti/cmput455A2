from board_base import (
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    opponent
)
from board import GoBoard
from board_util import GoBoardUtil
from typing import Any, Callable, Dict, List, Tuple
import math
import operator
from board_base import coord_to_point
INFINITY = 100000000000
seen_states = {}
import cProfile, pstats
from hasher import ZobristHash
from tt import TT

import signal
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    class TimeoutError(Exception):
        pass

def handler(signum, frame):
    raise TimeoutError()

    # set the timeout handler

def printMoves(list,board:GoBoard):
    print (" ".join([point_to_coord(word, board.size) for word in list]))

def callAlphabeta(rootState: GoBoard, timelimit):
    #print("Calling Alphabeta")
    copyboard = rootState.copy()
    hasher = ZobristHash(rootState.size)
    tt = TT()
    # result = alphabeta(rootState, copyboard,-INFINITY, INFINITY, 0, tt, hasher)
    # result = rootState.bestMoves()
    # printMoves(result, rootState)
    # return result
    ###### This is the profiling code ######
    # profiler = cProfile.Profile()
    # profiler.enable()
    # result = alphabeta(rootState, copyboard,-INFINITY, INFINITY, 0, tt, hasher)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()
    # return result
    ###### This is the final submission code #####
    def handler(s, f):
            raise TimeoutError("timedout")
    # signal.signal(signal.SIGALRM, handler) 
    # signal.alarm(int(timelimit))
    # #######################################################################################
    print(timelimit)
    try:
        signal.signal(signal.SIGALRM, handler) 
        signal.alarm(int(timelimit))
        result = alphabeta(rootState, copyboard,-INFINITY, INFINITY, 0, tt, hasher)
    except TimeoutError as exc:
        #print(exc)
        print("timedout")
        result = "unknown"
    finally:
        signal.alarm(0)
        return result
    #########################################################################################
    # faulty hash (just in case)
    """ list = GoBoardUtil.get_twoD_board(copy)
    exp = 0
    code = 0
    for item in list:
        for num in item:
            code += num*(math.pow(3,exp))
            exp+=1
    if(code in seen_states.keys()):
        if(seen_states[code] == (-2*INFINITY,0)):
            result = (-2*INFINITY,0)
            return result
        else:
            result,player = seen_states[code]
            if(player != copy.current_player):
                result = result*-1
            return  result,None
    seen_states[code] = (-2*INFINITY,0)
    """

def alphabeta(board: GoBoard,copy, alpha, beta, depth, tt: TT, hasher: ZobristHash):

    l = GoBoardUtil.get_twoD_board(copy)
    code = hasher.hash(l.flatten(),copy.current_player)
    result = tt.lookup(code)
    if result != None:
        return result
    
    if copy.end_of_game():
        result = (copy.staticallyEvaluateForToPlay(), None)
        tt.store(code, result)
        return result

    moves = copy.bestMoves()
    move = moves[0]
    
    for m in moves:
        if depth == 0:
            copy = board.copy()
            copy.printMoves(moves)
        _,cap = copy.play_move(m, copy.current_player)
        value,_ = alphabeta(board, copy, -beta, -alpha,depth+1, tt, hasher)
        value = -value
        if value > alpha:
            alpha = value
            move = m
        copy.undo_move(m,cap)
        #seen_states[code] = (value, copy.current_player)
        if alpha >= beta:
            result = (beta, point_to_coord(move, copy.size))
            tt.store(code, result)
            return result

    result = (alpha, point_to_coord(move, copy.size))
    tt.store(code, result)
    return result

def sample(board:GoBoard):
    five = board.detect_n_in_row(5)
    #four = board.detect_n_in_row(4)
    #three = board.detect_n_in_row(3)
    ordered_moves = list(dict.fromkeys(five))
    ordered_moves = list(dict.fromkeys(five+four+three))
    ordered_moves += list(set(board.legal_moves())-set(ordered_moves))
    return ordered_moves


'''def search_hash(board:GoBoard, tt: TT, hasher: ZobristHash, moves):
    wins = []
    others = []
    copy = board.copy()
    l = GoBoardUtil.get_twoD_board(copy)
    code = hasher.hash(l.flatten(),copy.current_player)
    for m in moves:
        _,cap = copy.play_move(m,copy.current_player)
        l = GoBoardUtil.get_twoD_board(copy)
        code = hasher.update_hash(code,m,l.flatten())
        result = tt.lookup(code)
        if result == INFINITY:
            wins.append(m)
        else:
            others.append(m)
        copy.undo_move(m,cap)
        l = GoBoardUtil.get_twoD_board(copy)
        code = hasher.update_hash(code,m,l.flatten())
    return wins + others'''

def search_captures(board:GoBoard):
    return True

def order_moves(board: GoBoard):
    '''
    No longer being used.
    '''
    cur_play = board.current_player
    ordered_moves = []

    black_4, white_4 = board.detect_n_in_a_row(4)

    black_3, white_3 = board.detect_n_in_a_row(3)

    black_2, white_2 = board.detect_n_in_a_row(2)

    if cur_play == BLACK:
        if len(list(black_4.keys())) > 0:
            auto_win = list(black_4.keys())
            return auto_win
        if len(black_3) != 0:
            if (max(black_3.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_win = [max(black_3.items(), key=operator.itemgetter(1))[0]]
                return auto_win
        if len(black_2) != 0:
            if (max(black_2.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_win = [max(black_2.items(), key=operator.itemgetter(1))[0]]
                return auto_win
        
        if len(list(white_4.keys())) > 0:
            auto_block = list(white_4.keys())
            return auto_block
        if len(white_3) != 0:
            if (max(white_3.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_block = [max(white_3.items(), key=operator.itemgetter(1))[0]]
                return auto_block
        if len(white_2) != 0: 
            if (max(white_2.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_block = [max(white_2.items(), key=operator.itemgetter(1))[0]]
                return auto_block

        ordered_moves += list(black_3.keys())
        
        ordered_moves += list(set(list(black_2.keys()))-set(ordered_moves))
        
        ordered_moves += list(set(board.legal_moves())-set(ordered_moves))
        
        return ordered_moves
    
    else:
        if len(list(white_4.keys())) > 0:
            auto_win = list(white_4.keys())
            return auto_win
        if len(white_3) != 0: 
            if (max(white_3.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_win = [max(white_3.items(), key=operator.itemgetter(1))[0]]
                return auto_win
        if len(white_2) != 0: 
            if (max(white_2.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_win = [max(white_2.items(), key=operator.itemgetter(1))[0]]
                return auto_win
        
        if len(list(black_4.keys())) > 0:
            auto_block = list(black_4.keys())
            return auto_block
        if len(black_3) != 0: 
            if (max(black_3.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_block = [max(black_3.items(), key=operator.itemgetter(1))[0]]
                return auto_block
        if len(black_2) != 0: 
            if (max(black_2.items(), key=operator.itemgetter(1))[1] >= 5):
                auto_block = [max(black_2.items(), key=operator.itemgetter(1))[0]]
                return auto_block

        ordered_moves += list(white_3.keys())

        ordered_moves += list(set(list(white_2.keys()))-set(ordered_moves))

        ordered_moves += list(set(board.legal_moves())-set(ordered_moves))

        return ordered_moves
    



####################################################################################################
#### Helper Functions from the code base this is just for testing, we shouldnt need them later #####
####################################################################################################
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

###################################
### Testing/Debugging #############
###################################
def undoTest(board: GoBoard):
    board.play_move(move_to_coord("a1", board.size),BLACK)
    board.play_move(move_to_coord("a2", board.size),WHITE)
    board.play_move(move_to_coord("a3", board.size),WHITE)
    print(GoBoardUtil.get_twoD_board(board))
    _, cap = board.play_move(move_to_coord("a4", board.size),BLACK)
    print(cap)
    print(GoBoardUtil.get_twoD_board(board))
    board.play_move(move_to_coord("b1", board.size),BLACK)
    board.play_move(move_to_coord("b2", board.size),WHITE)
    board.play_move(move_to_coord("b3", board.size),WHITE)
    print(GoBoardUtil.get_twoD_board(board))
    _, cap1 = board.play_move(move_to_coord("b4", board.size),BLACK)
    print(GoBoardUtil.get_twoD_board(board))
    board.undo_move(move_to_coord("b4", board.size), cap1)
    board.undo_move(move_to_coord("a4", board.size), cap)
    print(GoBoardUtil.get_twoD_board(board))