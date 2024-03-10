import numpy as np
import pygame
import time

ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARE_SIZE=100
width = COLUMN_COUNT * SQUARE_SIZE
height = (ROW_COUNT+1)*SQUARE_SIZE
SPEED = 40
size=(height, width)
RADIUS = int(SQUARE_SIZE/2 - 5)
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)



class Game:
  def __init__(self):
    self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    self.screen = pygame.display.set_mode(size) 
    self.clock = pygame.time.Clock()
    self.prev_state = self.board
    self.prev_action = None 
    self.player1_wins = 0
    self.player2_wins = 0
    self.player1_redo = 0
    self.player2_redo = 0
    self.num_games = 0
    # will remember the last state of the player was on before move was made
    # its new_state will be the old_state of the other player
    self.reset()
    
  def reset(self):
    self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    self.score = 0
    self.frame_iteration = 0
  
  def play_step(self, action, player):
    self.frame_iteration += 1

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()
        
    reward = 1
    game_over = False
    redo = False
    draw = False
    
    # check if the action is valid -> the column has place to put a new piece
    if not self.is_valid_move(action):
      reward = -10
      redo = True
      done = True
      return reward,game_over,self.score,redo,draw
      
    self.prev_state = self.board
    self.prev_action = action
    self._move(action, player)
    # updates the ui as well
    
    
    # check for game over
    # game over will have the other player winning the game
    # that will be done in a seperate function
    # for now we can check for draws -> no more space on board and having no winner
    if self.board_full():
      draw = True
      game_over = True
      reward = -10
      self.reset()
      return reward, game_over, self.score, redo, draw

    if self.win(player):
      game_over = True
      reward = 10
      self.draw_board()
      self.reset()
      
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
    self.draw_board()
  
  def _move(self, action, player):
      row = self.get_valid_row(action)
      self.drop_piece(row, action, player)
  
  def win(self, player):
    # Check horizontal locations for win
    board = self.board
    for c in range(COLUMN_COUNT-3):
      for r in range(ROW_COUNT):
        if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
          # board[r][c] = board[r][c+1] = board[r][c+2] = board[r][c+3] = 3
          return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
      for r in range(ROW_COUNT-3):
        if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
          # board[r][c] = board[r+1][c] = board[r+2][c] = board[r+3][c] = 3
          return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
      for r in range(ROW_COUNT-3):
        if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
          # board[r][c] = board[r+1][c+1] = board[r+2][c+2] = board[r+3][c+3] = 3
          return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
      for r in range(3, ROW_COUNT):
        if board[r][c] == player and board[r-1][c+1] == player and board[r-2][c+2] == player and board[r-3][c+3] == player:
          # board[r][c] = board[r-1][c-1] = board[r-2][c-2] = board[r-3][c-3] = 3
          return True
    
    return False
  
  def draw_board(self):
    
      for c in range(COLUMN_COUNT):
          for r in range(ROW_COUNT):
              pygame.draw.rect(self.screen, BLUE, (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
              pygame.draw.circle(self.screen, BLACK, (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)), RADIUS)

      for c in range(COLUMN_COUNT):
          for r in range(ROW_COUNT):
              if self.board[r][c] == 1:
                  pygame.draw.circle(self.screen, RED, (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), height - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)), RADIUS)
              elif self.board[r][c] == 2:
                  pygame.draw.circle(self.screen, YELLOW, (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), height - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)), RADIUS)
      font = pygame.font.SysFont("monospace", 20)
      player1_wins_label = font.render("Player 1 Wins: " + str(self.player1_wins), 1, RED, (0,0,0))
      self.screen.blit(player1_wins_label, (10, 10))

      # Display Player 2 wins
      player2_wins_label = font.render("Player 2 Wins: " + str(self.player2_wins), 1, YELLOW,(0,0,0))
      self.screen.blit(player2_wins_label, (width - 250, 10))

      num_games_label = font.render("Num Games: " + str(self.num_games), 1, RED, (0,0,0))
      self.screen.blit(num_games_label, (width - 450, 10))

      # Display Player 1 Redo
      player1_redo_label = font.render("Player 1 Redo: " + str(self.player1_redo), True, RED, (0,0,0))
      self.screen.blit(player1_redo_label, (10, 60))

      # Display Player 2 Redo
      player2_redo_label = font.render("Player 2 Redo: " + str(self.player2_redo), True, YELLOW, (0,0,0))
      self.screen.blit(player2_redo_label, (width - 250, 60))
      
      pygame.display.update()

    

      

  
      
      
      
    
    