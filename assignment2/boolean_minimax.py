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
from board_base import coord_to_point
INFINITY = 100000000000

# def minimaxBooleanOR(board: GoBoard):
#     # print("lastmove",board.last_move)
#     # print("current player", board.current_player)
#     if board.detect_five_in_a_row():
#         return GO_COLOR(board.current_player)
#     # print(moves)
#     # board.play_move(moves[0], GO_COLOR(board.current_player))
#     # print(GoBoardUtil.get_twoD_board(board))
#     # board.undo_move(moves[0])
#     # print(GoBoardUtil.get_twoD_board(board))
#     moves = board.legal_moves()
#     for m in moves:
#         board.play_move(m, GO_COLOR(board.current_player))
#         print(GoBoardUtil.get_twoD_board(board))
#         isWin = minimaxBooleanAND(board)
#         board.undo_move()
#         if isWin:
#             return True
#     return False

# def minimaxBooleanAND(board: GoBoard):
#     if board.detect_five_in_a_row():
#         return GO_COLOR(board.current_player)
#     moves = board.legal_moves()
#     for m in moves:
#         board.play_move(m, GO_COLOR(board.current_player))
#         print(GoBoardUtil.get_twoD_board(board))
#         isLoss = not minimaxBooleanOR(board)
#         board.undo_move()
#         if isLoss:
#             return False
#     return True


# 
# def alphabeta(board: GoBoard, alpha, beta, depth):
#     if board.end_of_game() or depth == 0:
#         return board.staticallyEvaluateForToPlay()
#     move = None
#     for m in board.legal_moves():
#         _,capture = board.play_move(m, GO_COLOR(board.current_player))
#         print(GoBoardUtil.get_twoD_board(board))
#         value = -alphabeta(board, -beta, -alpha, depth-1)
#         if value > alpha:
#             alpha = value
#         board.undo_move(m, capture)
#         if value >= beta: 
#             return beta # or value in failsoft (later)
#     return alpha

# initial call with full window

def callAlphabeta(rootState: GoBoard):
    copyboard = rootState.copy()
    return alphabeta(rootState,copyboard,-INFINITY, INFINITY, 0)

def alphabeta(board: GoBoard,copy, alpha, beta, depth):
    result = (0,0)
    if board.end_of_game():
        print("true", depth)
        result = (board.staticallyEvaluateForToPlay(), None)
        return result

    # when we have a move ordering function, add an if statement to check depth = 0 
    # if yes use the move ordering function else use the board.legalmoves
    moves = board.legal_moves()
    m = moves[0]
    # print(moves)
    
    for m in moves:
        if depth == 0:
            board = copy
        _,cap = board.play_move(m, board.current_player)
        # print(GoBoardUtil.get_twoD_board(board))
        value,_ = alphabeta(board,copy, -beta, -alpha,depth+1)
        value = -value
        if value > alpha:
            alpha = value
            m = m
        board.undo_move(m,cap)
        if value >= beta:
            result = (beta, point_to_coord(m, board.size))
            return result
    result = (alpha, point_to_coord(m, board.size))
    return result


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