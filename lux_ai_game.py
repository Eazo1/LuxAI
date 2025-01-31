# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:17:20 2025

@author: ahmad
"""

import random

class LuxAIGame:
    def __init__(self):
        self.board_size = 24  # 24x24 grid
        self.max_units = 16
        self.unit_move_cost = 1
        self.unit_sap_cost = 30
        self.unit_sap_range = 3
        self.unit_sensor_range = 2
        self.max_energy = 400
        self.start_energy = 100
        self.relic_reward_range = 2
        self.relic_positions = self.generate_relics()
        self.player_units = {1: self.spawn_units(1), 2: self.spawn_units(2)}
        self.player_scores = {1: 0, 2: 0}
    
    def generate_relics(self):
        """Generates random relic positions on the board."""
        relics = []
        for _ in range(6):
            x, y = random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)
            relics.append((x, y))
        return relics
    
    def spawn_units(self, player):
        """Spawns units for a player in their respective starting zone."""
        units = []
        start_x = 0 if player == 1 else self.board_size - 1
        for i in range(self.max_units):
            units.append({
                'id': i,
                'x': start_x,
                'y': i,
                'energy': self.start_energy
            })
        return units
    
    def move_unit(self, player, unit_id, direction):
        """Moves a unit in the specified direction (up, down, left, right)."""
        unit = self.player_units[player][unit_id]
        if unit['energy'] < self.unit_move_cost:
            print(f"Unit {unit_id} does not have enough energy to move.")
            return
        
        dx, dy = 0, 0
        if direction == 'up':
            dy = -1
        elif direction == 'down':
            dy = 1
        elif direction == 'left':
            dx = -1
        elif direction == 'right':
            dx = 1
        
        new_x, new_y = unit['x'] + dx, unit['y'] + dy
        if 0 <= new_x < self.board_size and 0 <= new_y < self.board_size:
            unit['x'], unit['y'] = new_x, new_y
            unit['energy'] -= self.unit_move_cost
            self.check_relic_collection(player, unit)
    
    def check_relic_collection(self, player, unit):
        """Checks if a unit is near a relic and awards points."""
        for relic in self.relic_positions:
            if abs(unit['x'] - relic[0]) <= self.relic_reward_range and abs(unit['y'] - relic[1]) <= self.relic_reward_range:
                self.player_scores[player] += 1
                print(f"Player {player} gained a relic point!")
                self.relic_positions.remove(relic)
                break
    
    def sap_action(self, player, unit_id, target_x, target_y):
        """Performs a sap action to drain energy from an enemy unit in range."""
        unit = self.player_units[player][unit_id]
        if unit['energy'] < self.unit_sap_cost:
            print(f"Unit {unit_id} does not have enough energy to sap.")
            return
        
        for enemy in self.player_units[3 - player]:  # Opposing player's units
            if abs(enemy['x'] - target_x) <= self.unit_sap_range and abs(enemy['y'] - target_y) <= self.unit_sap_range:
                enemy['energy'] = max(0, enemy['energy'] - self.unit_sap_cost)
                print(f"Player {player} sapped enemy unit at ({target_x}, {target_y})!")
                break
    
    def play_turn(self, player):
        """Allows a player to take a turn by moving or sapping."""
        print(f"Player {player}'s turn:")
        action = input("Enter action (move/sap): ")
        unit_id = int(input("Enter unit ID: "))
        
        if action == 'move':
            direction = input("Enter direction (up/down/left/right): ")
            self.move_unit(player, unit_id, direction)
        elif action == 'sap':
            target_x = int(input("Enter target x: "))
            target_y = int(input("Enter target y: "))
            self.sap_action(player, unit_id, target_x, target_y)
    
    def check_winner(self):
        """Checks the winner based on relic points."""
        if not self.relic_positions:
            if self.player_scores[1] > self.player_scores[2]:
                print("Player 1 wins!")
            elif self.player_scores[2] > self.player_scores[1]:
                print("Player 2 wins!")
            else:
                print("It's a tie!")
            return True
        return False
    
    def start_game(self):
        """Runs the game loop until all relics are collected."""
        while not self.check_winner():
            self.play_turn(1)
            if self.check_winner():
                break
            self.play_turn(2)

if __name__ == "__main__":
    game = LuxAIGame()
    game.start_game()
