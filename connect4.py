import numpy as np
import pygame

ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARE_SIZE=100
width = COLUMN_COUNT * SQUARE_SIZE
height = (ROW_COUNT+1)*SQUARE_SIZE
SPEED = 40


class Game:
  def __init__(self, player):
    self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    self.player = player
    self.screen = pygame.display.set_mode(size) 
    self.clock = pygame.time.Clock()
    self.prev_state = self.board
    self.prev_action = None 
    # will remember the last state of the player was on before move was made
    # its new_state will be the old_state of the other player
    self.reset()
    
  def reset(self):
    self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    self.score = 0
    self.frame_iteration = 0
  
  def play_step(self, action):
    self.frame_iteration += 1

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()
        
    reward = 0
    game_over = False
    redo = False
    draw = False
    
    # check if the action is valid -> the column has place to put a new piece
    if not self.is_valid_move(action):
      reward = -5
      redo = True
      return reward,game_over,self.score,redo,draw
      
    self.prev_state = self.board
    self.prev_action = action
    self._move(action)
    # updates the ui as well
    
    
    # check for game over
    # game over will have the other player winning the game
    # that will be done in a seperate function
    # for now we can check for draws -> no more space on board and having no winner
    if self.board_full():
      draw = True
      game_over = True
      reward = -10
      return reward, game_over, self.score, redo, draw

    if self.win():
      reward = 10
      
    self.clock.tick(SPEED)
    
    return reward, game_over, self.score, redo, draw
  
  def board_full(self):
    return not 0 in self.board
  
    
  def is_valid_move(self, move):
      return self.board[ROW_COUNT-1][move] == 0
    
  def get_valid_row(self, action):
    for r in range(ROW_COUNT):
  		if self.board[r][action] == 0:
			  return r
    
  def drop_piece(self, row, col, player):
    self.board[row][col] = player
  
  def _move(self, action):
      row = self.get_valid_row(action)
      self.drop_piece(row, action, self.player)
  
  def win(self):
    # Check horizontal locations for win
    piece = self.player
    for c in range(COLUMN_COUNT-3):
      for r in range(ROW_COUNT):
        if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
          return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
      for r in range(ROW_COUNT-3):
        if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
          return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
      for r in range(ROW_COUNT-3):
        if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
          return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
      for r in range(3, ROW_COUNT):
        if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
          return True
  

      

  
      
      
      
    
    