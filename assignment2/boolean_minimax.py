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

def minimaxBooleanOR(board: GoBoard):
    # print("lastmove",board.last_move)
    # print("current player", board.current_player)
    if board.detect_five_in_a_row():
        return GO_COLOR(board.current_player)
    # print(moves)
    # board.play_move(moves[0], GO_COLOR(board.current_player))
    # print(GoBoardUtil.get_twoD_board(board))
    # board.undo_move(moves[0])
    # print(GoBoardUtil.get_twoD_board(board))
    moves = board.legal_moves()
    for m in moves:
        board.play_move(m, GO_COLOR(board.current_player))
        print(GoBoardUtil.get_twoD_board(board))
        isWin = minimaxBooleanAND(board)
        board.undo_move()
        if isWin:
            return True
    return False

def minimaxBooleanAND(board: GoBoard):
    if board.detect_five_in_a_row():
        return GO_COLOR(board.current_player)
    moves = board.legal_moves()
    for m in moves:
        board.play_move(m, GO_COLOR(board.current_player))
        print(GoBoardUtil.get_twoD_board(board))
        isLoss = not minimaxBooleanOR(board)
        board.undo_move()
        if isLoss:
            return False
    return True


# INFINITY = 1000000
# def alphabeta(state, alpha, beta):
#     if state.endOfGame():
#         return state.staticallyEvaluateForToPlay() 
#     for m in state.legalMoves():
#         state.play(m)
#         value = -alphabeta(state, -beta, -alpha)
#         if value > alpha:
#             alpha = value
#         state.undoMove()
#         if value >= beta: 
#             return beta   # or value in failsoft (later)
#     return alpha

# # initial call with full window
# def callAlphabeta(rootState):
#     return alphabeta(rootState, -INFINITY, INFINITY)