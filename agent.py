import torch
from model import LinearQNet, QTrainer
from connect4 import Game
import random
from collections import deque
import numpy as np
import pygame


MAX_MEMORY = 100_00
BATCH_SIZE = 1000
LR = 0.001

class Agent:
  
  def __init__(self):
    self.num_games = 0
    self.epsilon = 1
    self.gamma = 0.9
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = LinearQNet((6,7), 256,7)
    self.trainer = QTrainer(self.model, lr=LR,gamma=self.gamma)
    self.prev_state = None
    self.prev_move = None
  

  def get_state(self,game):
    return game.board
  
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  
  
  def train_long(self):
    if len(self.memory) > BATCH_SIZE:
      random_sample = random.sammple(self.memory,
                                     BATCH_SIZE)
    random_sample = self.memory

    states, action, reward, next_states, done = zip(*random_sample)
    
    states = np.array(states).flatten()
    actions = list(actions)
    rewards = list(rewards)
    next_states = np.array(next_states).flatten()
    dones = list(dones)
    
    self.trainer.train_step(states, actions, rewards, next_states, dones)
  
  def train_short(self, state, action, reward, next_state, done):
    state = np.array(state).flatten()
    next_state = np.array(next_state).flatten()
    
    self.trainer.train_step(state, action, reward, next_state, done)
    

  def get_action(self, state):
    self.epsilon = 80 - self.num_games
    
    if random.randint(0,200) < self.epsilon:
      print("Random Move")
      move = random.randint(0,6)
    else:
      initial_state = torch.tensor(state,
                                   dtype=torch.float)
      pred = self.model(initial_state)
      idx = torch.argmax(pred).item()
      move = idx
      print(f"Predicted Move {move}")
    return move

def train():
    scores = []
    player1 = Agent()
    player2 = Agent()
    game = Game()
    turn = 1

    pygame.init()

    # RADIUS = int(SQUARESIZE/2 - 5)

    # screen = pygame.display.set_mode(size)
    game.draw_board()
    pygame.display.update()

    # train loop
    while True:
      # Player 1 goes first
      if turn == 1:
          old_state = player1.get_state(game)
          player1.prev_state = old_state
          move = player1.get_action(old_state)
          player1.prev_move = move
          reward, done, score, redo, draw = game.play_step(move,1) 
          new_state = player1.get_state(game)

          if redo:
            game.player1_redo += 1
            new_state = old_state
            game.board = old_state
            player1.num_games += 1
          
          
          # Update agent's memory and train
          player1.train_short(old_state, move, reward, new_state, done)
          player1.remember(old_state, move, reward, new_state, done)
          if redo % 200 == 0  and redo != 0:
            player1.train_long()

          if done:
            game.num_games += 1
            print("done 1")
            if draw:
              print("DRAW1")
            # the very prev move that p2 made that resulted in p1 making a move that led to p1's victory
            p2_reward = -10
            p2_old_state = player2.prev_state
            p2_action = player2.prev_move
            p2_new_state = old_state
            player2.remember(p2_old_state, action, p2_reward, p2_new_state, done)
            # remember for p2's loss
            game.reset()
            player1.num_games += 1
            game.player1_wins += 1
            player1.train_long()

      else:
          # Player 2 goes
          old_state = player2.get_state(game)
          player2.prev_state = old_state
          move = player2.get_action(old_state)
          player2.prev_move = move
          reward, done, score, redo, draw = game.play_step(move,2)
          new_state = player2.get_state(game)

          if redo:
            game.player2_redo += 1
            player2.num_games  += 1
            new_state = old_state
            game.board = old_state
          
          # Update agent's memory and train
          player2.train_short(old_state, move, reward, new_state, done)
          player2.remember(old_state, move, reward, new_state, done)
          if redo % 200 == 0 and redo != 0:
            player2.train_long()
          if done:
            game.num_games += 1
            print("done2")
            if draw:
              print("draw2")
            # the very prev move that p1 made that resulted in p2 making a move that led to p2's victory
            p1_reward = -10
            p1_old_state = player2.prev_state
            p1_action = player1.prev_move
            p1_new_state = old_state
            player1.remember(p1_old_state, action, p1_reward, p1_new_state, done)
            # remember for p1's loss
            game.reset()
            player2.num_games += 1
            game.player2_wins += 1
            player2.train_long()

      # Switch turns
      
      game.draw_board()
      
      turn = 3 - turn


if __name__ == "__main__":
  train()