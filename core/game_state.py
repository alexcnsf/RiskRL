import numpy as np
import random


class GameState:
    def __init__(self, num_territories=10, adjacency=None):
        self.state = np.zeros((num_territories, 2), dtype=int)
        self.remaining_troops = {1: 0, 2: 0}
        self.adjacency = adjacency or {} 

    def set_ownership(self, territory_id, player_id):
        self.state[territory_id, 0] = player_id

    def set_troops(self, territory_id, troop_count):
        self.state[territory_id, 1] = troop_count

    def get_owner(self, territory_id):
        return self.state[territory_id, 0]

    def get_troops(self, territory_id):
        return self.state[territory_id, 1]

    def get_owned_territories(self, player_id):
        return [i for i in range(len(self.state)) if self.state[i, 0] == player_id]

    def add_deployment_troops(self, player_id, num):
        self.remaining_troops[player_id] += num

    def deploy(self, player_id, territory_id, num_troops):
        if self.get_owner(territory_id) != player_id:
            raise ValueError(f"Player {player_id} does not own territory {territory_id}")
        if self.remaining_troops[player_id] < num_troops:
            raise ValueError(f"Not enough troops: {self.remaining_troops[player_id]} available")
        self.state[territory_id, 1] += num_troops
        self.remaining_troops[player_id] -= num_troops

    def get_state_vector(self):
        return self.state.flatten()

    def __str__(self):
            return (
                f"\nMap State:\n"
                f"  0({self.get_owner(0)}:{self.get_troops(0)}) -- 1({self.get_owner(1)}:{self.get_troops(1)}) -- 3({self.get_owner(3)}:{self.get_troops(3)}) -- 6({self.get_owner(6)}:{self.get_troops(6)})\n"
                f"   |          |\n"
                f"  2({self.get_owner(2)}:{self.get_troops(2)})    4({self.get_owner(4)}:{self.get_troops(4)}) -- 7({self.get_owner(7)}:{self.get_troops(7)})\n"
                f"   |                          |\n"
                f"  5({self.get_owner(5)}:{self.get_troops(5)}) -- 8({self.get_owner(8)}:{self.get_troops(8)})\n"
                f"                              |\n"
                f"                        9({self.get_owner(9)}:{self.get_troops(9)})\n"
            )

    '''
    def __str__(self):
        return "\n".join(
            f"Territory {i}: Owner={self.state[i,0]}, Troops={self.state[i,1]}"
            for i in range(len(self.state))
        )
    '''

    def can_attack(self, attacker_id, from_territory, to_territory):
        # Valid if:
        # - attacker owns from_territory
        # - doesn't own to_territory
        # - they are adjacent
        # - at least 2 troops in from_territory
        if self.get_owner(from_territory) != attacker_id:
            return False
        if self.get_owner(to_territory) == attacker_id:
            return False
        if to_territory not in self.adjacency.get(from_territory, []):
            return False
        if self.get_troops(from_territory) < 2:
            return False
        return True

    def attack(self, attacker_id, from_territory, to_territory, num_attack_troops):
        if not self.can_attack(attacker_id, from_territory, to_territory):
            raise ValueError("Illegal attack")

        from_troops = self.get_troops(from_territory)
        to_troops = self.get_troops(to_territory)

        max_allowed = min(3, from_troops - 1)
        if num_attack_troops < 1 or num_attack_troops > max_allowed:
            raise ValueError(f"Invalid number of attack troops: {num_attack_troops}, allowed: 1–{max_allowed}")

        atk_dice = sorted([random.randint(1, 6) for _ in range(num_attack_troops)], reverse=True)
        def_dice = sorted([random.randint(1, 6) for _ in range(min(2, to_troops))], reverse=True)

        atk_losses = 0
        def_losses = 0

        for a, d in zip(atk_dice, def_dice):
            if a > d:
                def_losses += 1
            else:
                atk_losses += 1

        self.state[from_territory, 1] -= atk_losses
        self.state[to_territory, 1] -= def_losses

        conquered = False
        if self.state[to_territory, 1] <= 0:
            self.set_ownership(to_territory, attacker_id)
            conquered = True

            # Move troops into conquered territory
            # For now, move same number as used to attack, or remaining troops, whichever is smaller
            move_troops = min(num_attack_troops, self.get_troops(from_territory) - 1)
            self.state[from_territory, 1] -= move_troops
            self.state[to_territory, 1] = move_troops

        return {
            "attacker_rolls": atk_dice,
            "defender_rolls": def_dice,
            "attacker_losses": atk_losses,
            "defender_losses": def_losses,
            "territory_conquered": conquered
        }