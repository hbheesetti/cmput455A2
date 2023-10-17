import random

class ZobristHash:
    def __init__(self, boardSize):
        self.index = boardSize * boardSize
        self.array = [[random.getrandbits(64) for j in range(3)]
                      for i in range(self.index)]
        #self.black_play = random.getrandbits(64)

    def hash(self, board,current_player):
        code = self.array[0][board[0]]
        for i in range(1, self.index):
            code = code ^ self.array[i][board[i]]
        #if(current_player == 1):
            #code = code ^ self.black_play
        return code

    '''def update_hash(self,code,move,board):
        code = code ^ self.black_play
        move = move-(board.size+1)+(int)(move/(board.size+1))
        code = code ^ self.array[move][board[move]]
        return code'''
    #move-(board.size+1)+(int)(move/(board.size+1))