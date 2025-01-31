# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:20:35 2025

@author: ahmad
"""

import pygame
from lux_ai_game import LuxAIGame

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 24
CELL_SIZE = WIDTH // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lux AI Game")

# Load game
game = LuxAIGame()
current_player = 1

# Function to draw the grid
def draw_grid():
    screen.fill(WHITE)
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y))
    
    # Draw relics
    for relic in game.relic_positions:
        pygame.draw.rect(screen, GREEN, (relic[0] * CELL_SIZE, relic[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw units
    for player, units in game.player_units.items():
        color = RED if player == 1 else BLUE
        for unit in units:
            pygame.draw.circle(screen, color, (unit['x'] * CELL_SIZE + CELL_SIZE//2, unit['y'] * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3)

# Main game loop
running = True
while running:
    draw_grid()
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            unit_id = 0  # For simplicity, controlling the first unit
            if event.key == pygame.K_UP:
                game.move_unit(current_player, unit_id, 'up')
            elif event.key == pygame.K_DOWN:
                game.move_unit(current_player, unit_id, 'down')
            elif event.key == pygame.K_LEFT:
                game.move_unit(current_player, unit_id, 'left')
            elif event.key == pygame.K_RIGHT:
                game.move_unit(current_player, unit_id, 'right')
            
            if game.check_winner():
                running = False
            
            # Switch turns
            current_player = 3 - current_player

pygame.quit()
