class GreedyBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_deployments(self, game_state):
        owned = game_state.get_owned_territories(self.player_id)
        if not owned:
            return []
        
        # Find territories that border enemy territories
        border_territories = []
        for t in owned:
            for adj in game_state.adjacency.get(t, []):
                if game_state.get_owner(adj) != self.player_id:
                    border_territories.append(t)
                    break
        
        if border_territories:
            # Deploy to the border territory with the most troops
            strongest_border = max(border_territories, key=lambda tid: game_state.get_troops(tid))
            return [strongest_border] * game_state.remaining_troops[self.player_id]
        else:
            # If no border territories, deploy to the territory closest to enemy
            return [owned[0]] * game_state.remaining_troops[self.player_id]

    def choose_attack(self, game_state):
        best_attack = None
        best_ratio = -float("inf")

        for from_t in game_state.get_owned_territories(self.player_id):
            from_troops = game_state.get_troops(from_t)
            if from_troops < 2:
                continue  # Can't attack

            for to_t in game_state.adjacency.get(from_t, []):
                if not game_state.can_attack(self.player_id, from_t, to_t):
                    continue

                to_troops = game_state.get_troops(to_t)
                # Attack if we have more troops (at least 1.2x)
                attack_ratio = (from_troops - 1) / (to_troops + 1)  # -1 because we need to leave one troop behind
                
                if attack_ratio > best_ratio and attack_ratio > 1.2:  # Lower threshold for attacking
                    best_ratio = attack_ratio
                    # Use more troops for attack (up to half of available troops)
                    num_troops = min(from_troops - 1, max(3, (from_troops - 1) // 2))
                    best_attack = (from_t, to_t, num_troops)

        return best_attack  # can be None

    def choose_fortify(self, game_state):
        owned = game_state.get_owned_territories(self.player_id)
        if len(owned) < 2:
            return None
            
        # Find territories that don't border enemies
        safe_territories = []
        border_territories = []
        
        for t in owned:
            is_border = False
            for adj in game_state.adjacency.get(t, []):
                if game_state.get_owner(adj) != self.player_id:
                    is_border = True
                    border_territories.append(t)
                    break
            if not is_border:
                safe_territories.append(t)
                
        if not safe_territories or not border_territories:
            return None
            
        # Find the territory with the most troops that isn't on the border
        source = max(safe_territories, key=lambda tid: game_state.get_troops(tid))
        if game_state.get_troops(source) < 2:
            return None
            
        # Move troops to the weakest border territory
        target = min(border_territories, key=lambda tid: game_state.get_troops(tid))
        troops_to_move = game_state.get_troops(source) - 1
        
        if troops_to_move > 0:
            return (source, target, troops_to_move)
        return None
