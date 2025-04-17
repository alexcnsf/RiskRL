import numpy as np
import random

class StupidBot:
    def __init__(self):
        pass
        
    def get_action(self, game_state):
        """Get action from StupidBot"""
        # Get all owned territories sorted by troop count (descending)
        owned_territories = game_state.get_owned_territories(2)  # Player 2 is StupidBot
        if not owned_territories:
            return None, None, None
            
        # Sort territories by troop count (descending)
        owned_territories.sort(key=lambda t: game_state.get_troops(t), reverse=True)
        
        # Try to attack from the territory with most troops
        for from_territory in owned_territories:
            # Get all possible enemy territories to attack
            possible_targets = []
            for to_territory in game_state.adjacency[from_territory]:
                if game_state.get_owner(to_territory) == 1:  # Enemy territory
                    possible_targets.append(to_territory)
            
            if possible_targets:
                # Choose random enemy territory to attack
                to_territory = random.choice(possible_targets)
                # Attack with all but 1 troop
                num_troops = game_state.get_troops(from_territory) - 1
                return from_territory, to_territory, num_troops
      
        # If no possible attacks, return None
        return None, None, None 