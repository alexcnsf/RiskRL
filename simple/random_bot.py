import numpy as np
import random

class RandomBot:
    def __init__(self):
        pass
        
    def get_action(self, game_state):
        """Get a random valid attack action"""
        # Get all possible attacks
        possible_attacks = []
        for from_territory in game_state.get_owned_territories(2):  # Player 2 is RandomBot
            for to_territory in game_state.adjacency[from_territory]:
                if game_state.get_owner(to_territory) == 1:  # Enemy territory
                    possible_attacks.append((from_territory, to_territory))
        
        if not possible_attacks:
            return None, None, None
            
        # Choose random attack
        from_territory, to_territory = random.choice(possible_attacks)
        
        # Attack with a random number of troops (must leave 1 behind)
        max_troops = game_state.get_troops(from_territory) - 1
        if max_troops <= 0:  # If we can't attack (only 1 troop)
            return None, None, None
            
        num_troops = random.randint(1, min(3, max_troops))  # Max 3 troops like in Risk
        
        return from_territory, to_territory, num_troops 