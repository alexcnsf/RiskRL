class GreedyBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_deployments(self, game_state):
        owned = game_state.get_owned_territories(self.player_id)
        if not owned:
            return []
        
        # Deploy all troops to territory with the most troops already
        strongest = max(owned, key=lambda tid: game_state.get_troops(tid))
        return [strongest] * game_state.remaining_troops[self.player_id]

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
                # Only attack if we have more troops (at least 1.5x)
                attack_ratio = (from_troops - 1) / (to_troops + 1)  # -1 because we need to leave one troop behind
                
                if attack_ratio > best_ratio and attack_ratio > 1.5:  # Only attack if we have significant advantage
                    best_ratio = attack_ratio
                    num_troops = min(3, from_troops - 1)
                    best_attack = (from_t, to_t, num_troops)

        return best_attack  # can be None

    def choose_fortify(self, game_state):
        return None  # skip for now
