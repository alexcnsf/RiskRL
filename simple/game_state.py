import numpy as np
import random

class SimpleGameState:
    def __init__(self):
        # Define adjacency map
        self.adjacency = {
            0: [1, 2, 3, 4],  # Territory 0 connected to 1,2,3,4
            1: [0, 2, 3, 4, 5],  # Territory 1 connected to 0,2,3,4,5
            2: [0, 1, 4, 5, 6],  # Territory 2 connected to 0,1,4,5,6
            3: [0, 1, 4, 6, 7],  # Territory 3 connected to 0,1,4,6,7
            4: [0, 1, 2, 3, 5, 7, 8],  # Territory 4 connected to 0,1,2,3,5,7,8
            5: [1, 2, 4, 6, 8, 9],  # Territory 5 connected to 1,2,4,6,8,9
            6: [2, 3, 5, 7, 9],  # Territory 6 connected to 2,3,5,7,9
            7: [3, 4, 6, 8, 9],  # Territory 7 connected to 3,4,6,8,9
            8: [4, 5, 7, 9],  # Territory 8 connected to 4,5,7,9
            9: [5, 6, 7, 8]  # Territory 9 connected to 5,6,7,8
        }
        
        # Initialize state as numpy array (10 territories, 2 features: owner and troops)
        self.state = np.zeros((10, 2))
        self.state[:5, 0] = 1  # Player 1 owns first 5 territories
        self.state[5:, 0] = 2  # Player 2 owns last 5 territories
        self.state[:, 1] = 5  # Start with 5 troops per territory
        
        self.current_player = 1
        self.turn = 1
        
    def get_owner(self, territory_id):
        return int(self.state[territory_id, 0])
        
    def get_troops(self, territory_id):
        return int(self.state[territory_id, 1])
        
    def get_owned_territories(self, player_id):
        return [i for i in range(10) if self.get_owner(i) == player_id]
        
    def add_deployment_troops(self, player_id, num):
        # Add troops to player's reserve
        self.state[player_id - 1, 1] += num
        
    def deploy(self, player_id, territory_id, num_troops):
        if self.get_owner(territory_id) != player_id:
            raise ValueError(f"Player {player_id} does not own territory {territory_id}")
        if self.state[player_id - 1, 1] < num_troops:
            raise ValueError(f"Not enough troops: {self.state[player_id - 1, 1]} available")
        self.state[territory_id, 1] += num_troops
        self.state[player_id - 1, 1] -= num_troops
        
    def get_frontline_territories(self, player_id):
        """Get territories that border enemy territories"""
        frontline = []
        for t in self.get_owned_territories(player_id):
            for adj in self.adjacency.get(t, []):
                if self.get_owner(adj) != player_id:
                    frontline.append(t)
                    break
        return frontline
        
    def get_non_frontline_territories(self, player_id):
        """Get owned territories that don't border enemies"""
        frontline = set(self.get_frontline_territories(player_id))
        return [t for t in self.get_owned_territories(player_id) if t not in frontline]
        
    def can_attack(self, attacker_id, from_territory, to_territory):
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
        
        # In Risk, you can attack with any number of troops (must leave 1 behind)
        if num_attack_troops < 1 or num_attack_troops > from_troops - 1:
            raise ValueError(f"Invalid number of attack troops: {num_attack_troops}, must leave at least 1 troop behind")
            
        # Attacker can use up to 3 dice (but can attack with more troops)
        num_attacker_dice = min(3, num_attack_troops)
        # Defender can use up to 2 dice
        num_defender_dice = min(2, to_troops)
        
        atk_dice = sorted([random.randint(1, 6) for _ in range(num_attacker_dice)], reverse=True)
        def_dice = sorted([random.randint(1, 6) for _ in range(num_defender_dice)], reverse=True)
        
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
            self.state[to_territory, 0] = attacker_id
            conquered = True
            
            # Move troops into conquered territory (must leave at least 1 behind)
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
        
    def get_state_vector(self):
        """Get state vector for PPO agent"""
        # Basic territory info
        territory_info = self.state[:, 1]  # Troop counts (10 dim)
        
        # Get ownership (1 for player 1, 0 for player 2)
        ownership_info = np.array([1 if self.get_owner(i) == 1 else 0 for i in range(10)])  # (10 dim)
        
        # Get turn flag (1 for player 1's turn, 0 for player 2's turn)
        turn_flag = np.array([1 if self.current_player == 1 else 0])  # (1 dim)
        
        # Add territory adjacency information (simplified)
        adjacency = np.zeros(10)  # (10 dim)
        for i in range(10):
            for adj in self.adjacency[i]:
                if self.get_owner(adj) != self.get_owner(i):
                    adjacency[i] = 1
                    break
        
        # Concatenate all features
        return np.concatenate([territory_info, ownership_info, turn_flag, adjacency])  # Total: 31 dim
        
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
        
    def next_turn(self):
        """Switch to the next player's turn"""
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
    def is_game_over(self):
        """Check if the game is over"""
        p1_territories = len(self.get_owned_territories(1))
        p2_territories = len(self.get_owned_territories(2))
        return p1_territories == 0 or p2_territories == 0
        
    def get_winner(self):
        """Get the winner of the game"""
        p1_territories = len(self.get_owned_territories(1))
        p2_territories = len(self.get_owned_territories(2))
        if p1_territories == 0:
            return 2
        elif p2_territories == 0:
            return 1
        return None
        
    def get_deployment_troops(self, player_id):
        """Calculate deployment troops based on territories controlled"""
        num_territories = len(self.get_owned_territories(player_id))
        return max(1, 2 * (num_territories - 5))  # More aggressive deployment formula 