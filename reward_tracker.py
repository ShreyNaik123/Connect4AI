import matplotlib.pyplot as plt
from IPython import display
import numpy as np

plt.ion()

def plot(reward):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Check if the input reward is a list or a numpy array
    if isinstance(reward, (list, np.ndarray)):
        x_values = reward
    else:
        x_values = [reward]
    
    # Plot histogram of x values
    plt.hist(x_values, bins=20, color='blue', alpha=0.7)
    
    plt.title('Histogram of Rewards')
    plt.xlabel('Reward Values')
    plt.ylabel('Frequency')
  
    plt.show(block=False)
    plt.pause(0.1)
