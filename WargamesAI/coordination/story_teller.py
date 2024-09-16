from typing import Optional, Dict, List
from WargamesAI.coordination import Umpire, Game, GameRunner
from WargamesAI.agents import Agent
from WargamesAI.utils.easyLLM import EasyLLM
from WargamesAI.utils import json_schemas


class StoryTeller:
    """
    StoryTeller is responsible for creating a narrative-based professional wargame matrix game, including 
    generating the game rules, players, and rounds based on the narrative.
    """

    def __init__(self, narrative: Optional[str] = None) -> None:
        """
        Initializes the StoryTeller class.

        :param narrative: Optional string to define the theme of the game narrative.
        """
        self._llm = EasyLLM(max_new_tokens=5000)
        self._narrative = self._create_narrative(narrative)
        self._rules = self._create_game_rules()
        self._players = self._create_players()
        self._rounds = self._create_rounds()

    def _create_narrative(self, prompt: Optional[str] = None) -> str:
        """
        Creates the narrative for the game.

        :param prompt: Optional string to define the theme of the narrative.
        :return: The generated narrative as a string.
        """
        if prompt is None:
            narrative_prompt = "Generate a professional wargame matrix games narrative based on any narrative or theme."
        else:
            narrative_prompt = f"Generate a professional wargame matrix games narrative based on the following theme: {prompt}."

        narrative = self._llm.ask_question(
            self._llm.generate_json_prompt(json_schemas.DefaultModel, narrative_prompt)
        )["RESPONSE"]

        return narrative

    def _create_game_rules(self) -> str:
        """
        Creates the rules for the game based on the narrative.

        :return: The generated game rules as a string.
        """
        rules_prompt = f"Generate a professional wargame matrix games rule book based on the following narrative: {self._narrative}"

        rules = self._llm.ask_question(
            self._llm.generate_json_prompt(json_schemas.DefaultModel, rules_prompt)
        )["RESPONSE"]

        return rules

    def _create_players(self) -> List[Dict]:
        """
        Creates the list of players for the game based on the narrative and rules.

        :return: A list of dictionaries representing the players.
        """
        prompt = f"Populate the schema. \n\n Narrative: \n{self._narrative}. \n\nRules: \n{self._rules}"

        agents = self._llm.ask_question(
            self._llm.generate_json_prompt(json_schemas.MultipleAgentsModel, prompt)
        )

        return agents

    def _create_rounds(self) -> List[Dict]:
        """
        Creates the rounds and turns for the game based on the narrative and players.

        :return: A list of dictionaries representing the rounds.
        """
        prompt = f"Based on the following narrative and player characters for this professional wargame matrix game, create several rounds and turns in those rounds. Narrative: {self._narrative}. Rules: {self._rules}. Players: {self._players}"

        rounds = self._llm.ask_question(
            self._llm.generate_json_prompt(json_schemas.RoundModel, prompt)
        )

        return rounds

    def generate_game(self) -> Game:
        """
        Generates the complete game including teams and agents based on the narrative, rules, and players.

        :return: An instance of the Game class representing the complete game.
        """
        game = Game(rounds=self._rounds, game_rules_text=f"{self._narrative}. {self._rules}")

        agents: Dict[str, List[Dict[str, Agent]]] = {}
        for agent in self._players:
            name = agent["name"]
            team = agent["team"]
            deployment_directive = agent["deployment_directive"]
            factions = agent["factions"]
            beliefs = agent["beliefs"]
            disposition = agent["disposition"]
            empathy = agent["empathy"]
            exercise_objectives = agent["exercise_objectives"]
            strategic_objectives = agent["strategic_objectives"]

            agent_instance = Agent(
                game,
                deployment_directive=deployment_directive,
                factions=factions,
                beliefs=beliefs,
                disposition=disposition,
                empathy=empathy,
                exercise_objectives=exercise_objectives,
                strategic_objectives=strategic_objectives,
                bio_folder="."
            )
            if team not in agents:
                agents[team] = [{name: agent_instance}]
            else:
                agents[team].append({name: agent_instance})

        for team, team_agents in agents.items():
            game.add_team(team, team_agents)

        return game
