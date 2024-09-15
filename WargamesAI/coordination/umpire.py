import random
from WargamesAI.utils.easyLLM import EasyLLM
from WargamesAI.utils.easyRAG import EasyRAG
from WargamesAI.utils import json_schemas

class Umpire:
    """
    Represents the umpire or game master, handling game logic and player interactions.
    """

    def __init__(
        self,
        game,
        model_name=None,
        max_tokens=2000,
    ):
        """
        Initializes the Umpire instance.

        Args:
            game: The game instance.
            model_name (str): Name of the language model to use.
            max_tokens (int): Maximum tokens for LLM responses.
        """
        self._game = game
        self.llm = EasyLLM(max_new_tokens=max_tokens, model_name=model_name)
        self.rag = EasyRAG()
        self.actions = []
        self._resource_tracker = {}

        if game._resources:
            self._resources = game._resources.copy()
        else:
            self._resources = None

        # Initialize the umpire in the LLM
        game_rules = self._game._game_rules_text
        initial_prompt = self.llm.generate_json_prompt(
            json_schemas.DefaultModel,
            f"You are an umpire/game master of the game with the following rules. Be ready to facilitate the game. Respond acknowledging this\n\n"
            f"Game Rules:\n'{game_rules}'",
        )
        self.llm.ask_question(initial_prompt)

    def roll_dice(self, highest, lowest=1, times=1):
        """
        Rolls dice and returns the total score.

        Args:
            highest (int): The highest possible roll on the die.
            lowest (int): The lowest possible roll (default 1).
            times (int): Number of dice to roll.

        Returns:
            int: The total score from the dice rolls.
        """
        return sum(random.randint(lowest, highest) for _ in range(times))

    def pick_card(self, number):
        """
        Draws a number of cards from the game's deck.

        Args:
            number (int): Number of cards to draw.

        Returns:
            list: The drawn cards.
        """
        if self._game._cards:
            drawn_cards  = []

            for i in range(0, number):
                if len(self._game._cards) > 1:
                    card = random.sample(self._game._cards, 1)
                    drawn_cards.append(card)
                    self._game._cards.remove(card)

            return drawn_cards
        else:
            return []

    def _perform_umpire_action(self, required_action):
        """
        Performs an action as the umpire.

        Args:
            required_action (str): The action to perform.

        Returns:
            The response from the umpire's action.
        """
        main_query = (
            f"Following the rules of this game, perform the following action as Umpire: '{required_action}'. "
            f"The game state is: {self.get_game_status()}."
        )

        # Check if dice are required
        if self._game._use_dice:
            query = (
                f"Based on the current turn, is the use of a dice required by the Umpire? Current turn: {required_action}."
            )
            resp = self.llm.ask_question(
                self.llm.generate_json_prompt(json_schemas.SystemUseModel, query)
            )
            if resp.get("ITEM") == "DICE":
                dice = resp["ACTION"]
                number_of_dice, dice_sides = map(int, dice.split("d"))
                result = self.roll_dice(times=number_of_dice, highest=dice_sides)
                main_query += f" A '{dice}' was rolled and the result was '{result}'."

        # Check if cards are required
        if self._game._cards:
            query = (
                f"Based on the current turn, is the use of drawing a random card required by the Umpire? Current turn: {required_action}."
            )
            resp = self.llm.ask_question(
                self.llm.generate_json_prompt(json_schemas.SystemUseModel, query)
            )
            if resp.get("ITEM") == "CARD":
                number = int(resp["ACTION"])
                result = self.pick_card(number)
                main_query += f" {number} cards were drawn with the following results: {result}."

        # Get the umpire's action response
        resp = self.llm.ask_question(
            self.llm.generate_json_prompt(json_schemas.ActionResponseModel, main_query)
        )

        # Check legality
        is_legal = self._check_legality_of_action(resp)
        max_attempts = 5
        attempt = 1

        while not is_legal and attempt < max_attempts:
            new_prompt = (
                f"Your last action was deemed not legal in the game rules. The game rules are: "
                f"{self._game._game_rules_text}. Try again. {main_query}"
            )
            resp = self.llm.ask_question(
                self.llm.generate_json_prompt(json_schemas.ActionResponseModel, new_prompt)
            )
            is_legal = self._check_legality_of_action(resp)
            attempt += 1

        if not is_legal:
            raise Exception("Game failed. LLM couldn't follow the game rules.")

        self.actions.append(resp)
        return resp

    def ask_human_player_for_action(self, action):
        """
        Asks a human player for their action.

        Args:
            action (str): The action required.

        Returns:
            The player's response.
        """
        prompt = (
            f"It is your turn in the game. You are being presented with this action: '{action}'. "
            f"Return your response in the following format: '{json_schemas.ACTION_RESPONSE_JSON_SCHEMA}'"
        )
        response = input(prompt)

        is_legal = self._check_legality_of_action(response)
        max_attempts = 5
        attempt = 1

        while not is_legal and attempt < max_attempts:
            new_prompt = (
                f"Your last action was deemed not legal in the game rules. The game rules are: "
                f"{self._game._game_rules_text}. Try again."
            )
            response = input(new_prompt)
            is_legal = self._check_legality_of_action(response)
            attempt += 1

        if not is_legal:
            return False  # Player failed to make legal move

        self.actions.append(response)
        return response

    def _ask_player_for_action(self, team, player, action):
        """
        Asks an AI player for their action.

        Args:
            team (str): The player's team.
            player (str): The player's name.
            action (str): The action required.

        Returns:
            The player's response.
        """
        prompt = f"It is your turn in the game. {action}"
        agent = self._game._teams[team][player]
        resp = agent.request_action(prompt)

        is_legal = self._check_legality_of_action(resp)
        max_attempts = 5
        attempt = 1

        while not is_legal and attempt < max_attempts:
            new_prompt = (
                f"Your last action was deemed not legal in the game rules. The game rules are: "
                f"{self._game._game_rules_text}. Try again. {prompt}"
            )
            resp = agent.request_action(new_prompt)
            is_legal = self._check_legality_of_action(resp)
            attempt += 1

        if not is_legal:
            return False  # Player failed to make legal move

        self.actions.append(resp)
        return resp

    def engage_turn(self, turn):
        """
        Engages a turn in the game.

        Args:
            turn (dict): The turn information.

        Returns:
            The responses resulting from the turn.
        """
        player_team = turn["TEAM"]
        target_player = turn["PLAYER"]
        required_action = turn["ACTIVITY"]
        
        if target_player == "Umpire":
            response = self._perform_umpire_action(required_action)

        else:

            teams = self._game._teams
            if player_team in teams:
                if target_player in teams[player_team]:
                    if teams[player_team][target_player]._is_human:
                        response = self.ask_human_player_for_action(required_action)
                        return
                    else:
                        response = self._ask_player_for_action(player_team, target_player, required_action)
                        return
            raise Exception (f"Agent/ player {target_player} of team {player_team} not found!")
        

        # Handle resources if applicable
        #if self._resource_tracker:
        #    resources = response.get("RESOURCES", [])
        #    for resource in resources:
        #        name = resource["NAME"]
        #        modification = resource["MODIFIER"]
        #        modifier = modification[:1]
        #        number = int(modification[1:])
        #        if modifier == "+":
        #            self.add_resource(player_team, target_player, name, number)
        #        elif modifier == "-":
        #            self.subtract_resource(player_team, target_player, name, number)
        #        else:
        #            raise ValueError(f"Invalid modifier: {modifier} in resource: {resource}.")

        final_responses = [response]
        #further_actions = self._check_response(player_team, target_player, required_action, response)
        #if isinstance(further_actions, list):
        #    for action in further_actions:
        #        final_responses.extend(self.engage_turn(action))

        return final_responses

    def _check_response(self, team, player, action, response):
        """
        Checks if there are further actions required based on a player's response.

        Args:
            team (str): The player's team.
            player (str): The player's name.
            action (str): The action performed.
            response (str): The player's response.

        Returns:
            Any further actions required.
        """
        query = (
            f"The player '{player}' of team '{team}' was asked to perform the following: '{action}'. "
            f"They responded with '{response}'. In line with the game rules, are there any follow-on actions required?"
        )
        further_actions = self.llm.ask_question(
            self.llm.generate_json_prompt(json_schemas.TurnModel, query)
        )
        return further_actions

    def _check_legality_of_action(self, action):
        """
        Checks if an action is legal according to the game rules.

        Args:
            action: The action to check.

        Returns:
            bool: True if legal, False otherwise.
        """
        query = (
            f"Does the following action follow the rules of the game? {action}. "
            f"The game state is: {self.get_game_status()}."
        )
        return bool(self.rag.ask_question_with_pdf(query, self._game._game_rules_pdf))

    def get_game_status(self):
        """
        Gets the current game status.

        Returns:
            dict: The game status.
        """
        status = {
            "actions": self.actions,
            "teams": self._game._teams,
            "resources": self._resource_tracker,
        }
        return status

    def add_resource(self, team_name, player_name, resource, number):
        """
        Adds a resource to a player.

        Args:
            team_name (str): The team name.
            player_name (str): The player's name.
            resource (str): The resource name.
            number (int): The amount to add.

        Returns:
            The new resource amount.
        """
        self._resource_tracker.setdefault(team_name, {}).setdefault(player_name, {}).setdefault(resource, 0)
        self._resource_tracker[team_name][player_name][resource] += number
        return self._resource_tracker[team_name][player_name][resource]

    def subtract_resource(self, team_name, player_name, resource, number):
        """
        Subtracts a resource from a player.

        Args:
            team_name (str): The team name.
            player_name (str): The player's name.
            resource (str): The resource name.
            number (int): The amount to subtract.

        Returns:
            The new resource amount.
        """
        self._resource_tracker.setdefault(team_name, {}).setdefault(player_name, {}).setdefault(resource, 0)
        self._resource_tracker[team_name][player_name][resource] -= number
        return self._resource_tracker[team_name][player_name][resource]

    def produce_summary(self):
        actions = self.actions
        rules = self._game.game_rules_text
        players = self._game.teams
        prompt = self.llm.generate_json_prompt(json_schemas.DefaultModel, f"Based on the below game actions and the following game rules and players, summarise the game. \n\n Actions: \n {actions}. \n \n Rules: {rules} \n \n Players: \n {players}")
        response = self.llm.ask_question(prompt)["RESPONSE"]

        return response
    
    def deduce_winner(self):
        actions = self.actions
        rules = self._game.game_rules_text
        players = self._game.teams
        prompt = self.llm.generate_json_prompt(json_schemas.WinModel, f"Based on the below game actions and the following game rules and players, Deduce the winner of the game. \n\n Actions: \n {actions}. \n \n Rules: {rules} \n \n Players: \n {players}")
        response = self.llm.ask_question(prompt)

        return response
    