import os
from WargamesAI.utils import pdf_utils

class Game:
    """
    Represents a game, containing rules, rounds, teams, and other game elements.
    """

    def __init__(
        self,
        rounds,
        use_dice=False,
        cards=None,
        resources=None,
        game_rules_text=None,
        game_rules_pdf=None,
        teams=None,
        rules_folder=".",
    ):
        """
        Initializes the Game instance.

        Args:
            rounds: The rounds of the game.
            use_dice (bool): Whether dice are used in the game.
            cards: The cards used in the game.
            resources: The resources in the game.
            game_rules_text (str): The text of the game rules.
            game_rules_pdf (str): The path to the game rules PDF.
            max_turns: The maximum number of turns.
            teams (dict): The teams in the game.
            rules_folder (str): Folder to store the rules PDFs.
        """
        if (game_rules_pdf is None and game_rules_text is None) or (game_rules_text and game_rules_pdf):
            raise ValueError("Provide either 'game_rules_text' or 'game_rules_pdf', but not both.")

        if not os.path.exists(rules_folder):
            os.mkdir(rules_folder)

        if game_rules_text:
            hashed_rules_name = pdf_utils.hash_string(game_rules_text)
            self._game_rules_pdf = os.path.join(rules_folder, f"{hashed_rules_name}.pdf")
            pdf_utils.write_pdf(game_rules_text, self._game_rules_pdf)
            self._game_rules_text = game_rules_text
        elif game_rules_pdf:
            self._game_rules_pdf = game_rules_pdf
            self._game_rules_text = pdf_utils.read_pdf(game_rules_pdf)
        else:
            raise ValueError("Invalid game rules state!")

        self._rounds = rounds
        self._teams = teams if teams is not None else {}

        self._cards = cards
        self._resources = resources
        self._use_dice = use_dice

    def add_team(self, team_name, actors):
        """
        Adds a team to the game.

        Args:
            team_name (str): The name of the team.
            actors (list): The list of actors (agents) in the team.
        """
        if team_name in self._teams:
            raise ValueError(f"Team '{team_name}' already exists!")
        self._teams[team_name] = actors

    @property
    def teams(self):
        """Returns a copy of the teams dictionary."""
        return self._teams.copy()
