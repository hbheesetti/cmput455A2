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

def callAlphabeta(rootState: GoBoard, timelimit):
    #print("Calling Alphabeta")
    copyboard = rootState.copy()
    hasher = ZobristHash(rootState.size)
    tt = TT()
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
    result = "unknown"
    try:
        signal.signal(signal.SIGALRM, handler) 
        signal.alarm(int(timelimit))
        result = alphabeta(rootState, copyboard,-INFINITY, INFINITY, 0, tt, hasher)
    except TimeoutError as exc:
        result = "unknown"
    finally:
        signal.alarm(0)
        return result

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

    moves = sample(copy)
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
            result = (beta, move)
            tt.store(code, result)
            return result
    result = (alpha, move)
    tt.store(code, result)
    return result

def sample(board:GoBoard):
    five = board.detect_n_in_row(5)
    #four = board.detect_n_in_row(4)
    #three = board.detect_n_in_row(3)
    ordered_moves = list(dict.fromkeys(five))
    #ordered_moves = list(dict.fromkeys(five+four+three))
    ordered_moves += list(set(board.legal_moves())-set(ordered_moves))
    return ordered_moves