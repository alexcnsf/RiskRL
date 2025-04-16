class DefensiveBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_deployments(self, game_state):
        owned = game_state.get_owned_territories(self.player_id)
        if not owned:
            return []

        # Deploy to the territory with the fewest troops
        weakest = min(owned, key=lambda t: game_state.get_troops(t))
        return [weakest] * game_state.remaining_troops[self.player_id]

    def choose_attack(self, game_state):
        best_attack = None
        max_advantage = -float("inf")

        for from_t in game_state.get_owned_territories(self.player_id):
            from_troops = game_state.get_troops(from_t)
            if from_troops < 3:
                continue  # Too weak to risk attacking

            for to_t in game_state.adjacency.get(from_t, []):
                if not game_state.can_attack(self.player_id, from_t, to_t):
                    continue

                to_troops = game_state.get_troops(to_t)
                advantage = from_troops - to_troops

                # Only attack if troop advantage is large
                if advantage >= 3 and advantage > max_advantage:
                    max_advantage = advantage
                    num_attack_troops = min(3, from_troops - 1)
                    best_attack = (from_t, to_t, num_attack_troops)

        return best_attack  # could be None if no safe attack

    def choose_fortify(self, game_state):
        return None  # not implemented yet
