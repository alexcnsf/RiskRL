import random

class StupidBot:
    def __init__(self, player_id):
        self.player_id = player_id
        self.favorite_territory = None  # Always deploys to the same territory
        
    def choose_deployments(self, game_state):
        """Always deploy all troops to the first owned territory"""
        owned = list(game_state.get_owned_territories(self.player_id))
        if not owned:
            return []
            
        # Always pick the same territory to deploy to
        if self.favorite_territory is None or self.favorite_territory not in owned:
            self.favorite_territory = owned[0]
            
        return [self.favorite_territory] * game_state.remaining_troops[self.player_id]
        
    def choose_attack(self, game_state):
        """Never attack"""
        return None
        
    def choose_fortify(self, game_state):
        """Never fortify"""
        return None 