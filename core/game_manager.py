import numpy as np

class GameManager:
    def __init__(self, game_state, agent1, agent2):
        self.state = game_state
        self.agents = {1: agent1, 2: agent2}
        self.current_player = 1

    def give_deployment_troops(self, player_id):
        num_owned = len(self.state.get_owned_territories(player_id))
        troops = max(0, num_owned // 2)
        self.state.add_deployment_troops(player_id, troops)


    def run_turn(self):
        agent = self.agents[self.current_player]
        
        # 1. Deployment
        self.give_deployment_troops(self.current_player)
        deploy_choices = agent.choose_deployments(self.state)
        for territory_id in deploy_choices:
            self.state.deploy(self.current_player, territory_id, 1)

        # 2. Attack (1 random attack only for now)
        # NEW: attack as long as the bot wants
        while True:
            attack_choice = agent.choose_attack(self.state)
            if not attack_choice:
                break  # no more valid attacks

            from_t, to_t, num = attack_choice
            try:
                result = self.state.attack(self.current_player, from_t, to_t, num)
                #print(f"Player {self.current_player} attacks {from_t} → {to_t} | {result}")
            except ValueError as e:
                #print(f"Invalid attack skipped: {e}")
                break  # stop attacking after invalid attempt (optional)


        # 3. Fortify (not implemented)
        # fortify_choice = agent.choose_fortify(self.state)

        # 4. Switch player
        self.current_player = 2 if self.current_player == 1 else 1

    def is_game_over(self):
        owners = self.state.state[:, 0]
        return np.all(owners == 1) or np.all(owners == 2)

    def get_winner(self):
        return int(np.all(self.state.state[:, 0] == 1)) or 2