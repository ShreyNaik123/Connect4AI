import torch
from model import LinearQNet, QTrainer
from connect4 import Game
import random
from collections import deque


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
    self.trainer = QTrainer(self.model, lr=LR,
                            gamma=self.gamma
                            )
  

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
    
    states = list(states)
    actions = list(actions)
    rewards = list(rewards)
    next_states = list(next_states)
    dones = list(dones)
    
    self.trainer.train_step(states, actions, rewards, next_states, dones)
  
  def train_short(self, state):
    
    self.trainer.train_step(state, action, reward, next_state, done)
    

  def get_action(self, state):
    self.epsilon = 80 - self.num_games
    
    if random.randint(0,200) < self.epsilon:
      move = random.randint(0,6)
    else:
      initial_state = torch.tensor(state,
                                   dtype=torch.float)
      pred = self.model(initial_state)
      idx = torch.argmax(pred).item()
      move = idx
    return move

def train():
    scores = []
    agent = Agent()
    player1 = Game(1)
    player2 = Game(2)
    turn = 1

    # train loop
    while True:
      # Player 1 goes first
      if turn == 1:
          old_state = agent.get_state(player1)
          move = agent.get_action(old_state)
          reward, done, score, redo, draw = player1.play_step(move)
          new_state = agent.get_state(player1)

          if redo:
            new_state = old_state
          
          # Update agent's memory and train
          agent.train_short(old_state, move, reward, new_state, done)
          agent.remember(old_state, move, reward, new_state, done)

          if done:
            # the very prev move that p2 made that resulted in p1 making a move that led to p1's victory
            p2_reward = -10
            p2_old_state = player2.prev_state
            p2_action = player2.action
            p2_new_state = old_state
            agent.remember(p2_old_state, action, p2_reward, p2_new_state, done)
            # remember for p2's loss
            player1.reset()
            agent.num_games += 1
            agent.train_long()

      else:
          # Player 2 goes
          old_state = agent.get_state(player1)
          move = agent.get_action(old_state)
          reward, done, score, redo, draw = player2.play_step(move)
          new_state = agent.get_state(player2)

          if redo:
            new_state = old_state
          
          # Update agent's memory and train
          agent.train_short(old_state, move, reward, new_state, done)
          agent.remember(old_state, move, reward, new_state, done)

          if done:
            # the very prev move that p2 made that resulted in p1 making a move that led to p1's victory
            p1_reward = -10
            p1_old_state = player2.prev_state
            p1_action = player2.action
            p1_new_state = old_state
            agent.remember(p1_old_state, action, p1_reward, p1_new_state, done)
            # remember for p2's loss
            player2.reset()
            agent.num_games += 1
            agent.train_long()

      # Switch turns
      turn = 3 - turn


if __name__ == "__main__":
  train()