#Tic-Tac-Toe
import numpy as np
import random

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1

# Initialize Q-table
Q = {}

# Function to print the Tic-Tac-Toe board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("---------")
# Function to check if someone has won
def check_win(player, board):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or \
           all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or \
       all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

# Function to get the Q-value for a state-action pair
def get_Q_value(state, action):
    if (state, action) not in Q:
        Q[(state, action)] = 0
    return Q[(state, action)]

# Function to choose an action using epsilon-greedy strategy
def choose_action(state, legal_actions):
    if random.uniform(0, 1) < exploration_rate:
        # Explore: choose a random action
        return random.choice(legal_actions)
    else:
        # Exploit: choose the action with the highest Q-value
        Q_values = [get_Q_value(state, action) for action in legal_actions]
        return legal_actions[np.argmax(Q_values)]
# Function to update the Q-value using the Q-learning update rule
def update_Q_value(state, action, reward, next_state, next_legal_actions):
    best_next_action = max(next_legal_actions, key=lambda a: get_Q_value(next_state, a))
    Q[state, action] = (1 - learning_rate) * get_Q_value(state, action) + \
                       learning_rate * (reward + discount_factor * get_Q_value(next_state, best_next_action))

# Function to simulate a Tic-Tac-Toe game
def play_game():
    # Initialize the game board
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'  # Player 'X' is the human player, 'O' is the computer

    while True:
        print_board(board)

        if current_player == 'X':
            # Player 'X' (human) move
            while True:
                move = int(input("Enter your move (1-9): ")) - 1
                row, col = divmod(move, 3)
                if board[row][col] == ' ':
                    break
                else:
                    print("Invalid move. Try again.")
        else:
            # Player 'O' (computer) move
            legal_actions = [i for i in range(9) if board[i // 3][i % 3] == ' ']
            move = choose_action(tuple(tuple(row) for row in board), legal_actions)
            row, col = divmod(move, 3)

        # Update the board
        board[row][col] = current_player
        # Check if the game is over
        if check_win(current_player, board):
            print_board(board)
            print(f"Player '{current_player}' wins!")
            break
        elif all(val != ' ' for row in board for val in row):
            print_board(board)
            print("It's a tie!")
            break

        # Switch to the next player
        current_player = 'O' if current_player == 'X' else 'X'

if __name__ == "__main__":
    play_game()
