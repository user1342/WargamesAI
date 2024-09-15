from WargamesAI.coordination import Umpire, Game, GameRunner
from WargamesAI.agents import Agent
from WargamesAI.utils.easyLLM import EasyLLM
from WargamesAI.utils import json_schemas
#Help me make random people? 
# Build a game based on this premise, these win confitions, paramiters, actors, teams, game style, etc.

class StoryTeller():
    
    def __init__(self, narrative=None) -> None:
        self._llm = EasyLLM()
        self._narrative = self._create_narrative(narrative)
        self._rules = self._create_game_rules()
        self._players = self._create_players()
        self._rounds = self._create_rounds()

    def _create_narrative(self, prompt = None):
        narrative_prompt = ""
        if prompt == None:
            narrative_prompt = "Generate a professional wargame matrix games narrative based on any narrative or theme."
        else:
            narrative_prompt = f"Generate a professional wargame matrix games narrative based on the following theme: {prompt}."

        narrative = self._llm.ask_question(self._llm.generate_json_prompt(json_schemas.DefaultModel, narrative_prompt))["RESPONSE"]

        return narrative
    
    def _create_game_rules(self):
        rules_prompt = f"Generate a professional wargame matrix games rule book based on the following narrative: {self._narrative}"

        rules = self._llm.ask_question(self._llm.generate_json_prompt(json_schemas.DefaultModel, rules_prompt))["RESPONSE"]

        return rules

    def _create_players(self):
        prompt = f"Based on the following professional wargame matrix game narrative and rules create a list of player characters for the game. Narrative: {self._narrative}. Rules: {self._rules}"
        agents = self._llm.ask_question(self._llm.generate_json_prompt(json_schemas.MultipleAgentsModel, prompt))
        return agents

    def _create_rounds(self):
        prompt = "Based on the following narrative and player characters for this professional wargame matrix game, create several rounds and turns in those rounds.Narrative: {self._narrative}. Rules: {self._rules}. Players {self._players}"
        rounds = self._llm.ask_question(self._llm.generate_json_prompt(json_schemas.RoundModel, prompt))
        return rounds
    
    def generate_game(self):
        game = Game(rounds=self._rounds, game_rules_text=f"{self._narrative}. {self._rules}")
        
        agents = {}
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

            agent = Agent(game, deployment_directive=deployment_directive, factions=factions, beliefs=beliefs, disposition=disposition, empathy=empathy, exercise_objectives=exercise_objectives, strategic_objectives=strategic_objectives, bio_folder=".")
            if team not in agents:
                agents[team] = [{name:agent}]
            else:
                agents[team].append({name:agent})

        for team in agents:
            game.add_team(team, agents[team])

        return game
