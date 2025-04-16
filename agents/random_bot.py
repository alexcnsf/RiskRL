import random

class RandomBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_deployments(self, game_state):
        owned = game_state.get_owned_territories(self.player_id)
        return random.choices(owned, k=game_state.remaining_troops[self.player_id])

    def choose_attack(self, game_state):
        legal_attacks = []
        for from_t in game_state.get_owned_territories(self.player_id):
            for to_t in game_state.adjacency.get(from_t, []):
                if game_state.can_attack(self.player_id, from_t, to_t):
                    for n in range(1, min(3, game_state.get_troops(from_t) - 1) + 1):
                        legal_attacks.append((from_t, to_t, n))
        return random.choice(legal_attacks) if legal_attacks else None

    def choose_fortify(self, game_state):
        return None  # skip for now
