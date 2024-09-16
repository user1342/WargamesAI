import json 
from pydantic import BaseModel, Field, create_model
from typing import Any, Dict, List, Union, Type
from WargamesAI.utils.easyLLM import EasyLLM

# Corrected DEFAULT_RESPONSE with the closing parenthesis
DEFAULT_RESPONSE = json.dumps({"RESPONSE": "The response to the query"})

# Used when requesting resources be modified
RESOURCE_CHANGE_SCHEMA = [
    {
        "NAME": "The name of the resource being used. Resource use is optional.",
        "MODIFIER": "The number of that resource being used/ added use +/-. Resource use is optional."
    }
]

# Schema for response after a player/umpire has performed an action
ACTION_RESPONSE_JSON_SCHEMA = json.dumps({
    "ACTION": "The 'thing' you are doing/performing",
    "RATIONALE": "The reason why you are doing the requested action",
    #"RESOURCES": RESOURCE_CHANGE_SCHEMA,
    "TARGETS": "A list of players who are the targets of this action. Can be 'Umpire' for umpire/game master."
})

# Schema for a turn in a round 
TURN_JSON_SCHEMA = json.dumps({
    "TEAM": "The team the target player is in. Provide None if Umpire.", 


    "PLAYER": "The name of the player required to perform the action/activity/scenario. Use 'Umpire' if an action for the umpire/game master.",
    "ACTIVITY": "The thing required of the player"
})

# JSON schema for a round 
ROUND_JSON_SCHEMA = json.dumps([
    f"A list of turn dictionaries which follow the format: {TURN_JSON_SCHEMA}"
])

# Used by the LLM when requiring a dice or card draw
REQUIRES_SYSTEM_USE = json.dumps([{
    "ITEM": "CARD or DICE",
    "ACTION": "Dice sides for dice (e.g. 1d2,2d6,3d4) or number of cards for CARD"
}])

AGENT_REQUIREMENT_SCHEMA = json.dumps([{
    "player_character_name":"The name of the player character",
    "team":"The team the player character belongs to",
    "deployment_directive": "What the player characters overall situation is",
    "factions": ["A list of factions the player character belongs to"],
    "beliefs": ["A list of beliefs the player character holds"],
    "disposition": "Disposition of the player character",
    "empathy": "The level of empathy the player character has",
    "exercise_objectives": ["A list of short term objectives for this game"],
    "strategic_objectives": ["A list of long term objectives for this game or beyond"]
}])

LIST_OF_AGENTS_SCHEMA = json.dumps([{
    "player_character_name":"The name of the player character",
    "team":"The team the player character belongs to",
    "deployment_directive": "What the player characters overall situation is",
    "factions": ["A list of factions the player character belongs to"],
    "beliefs": ["A list of beliefs the player character holds"],
    "disposition": "Disposition of the player character",
    "empathy": "The level of empathy the player character has",
    "exercise_objectives": ["A list of short term objectives for this game"],
    "strategic_objectives": ["A list of long term objectives for this game or beyond"]
}])

WIN_SCHEMA = json.dumps({"WINNING_TEAN":"The name of the winning team","WINNING_PLAYER":"None if a Team won, though, player name if a specific player in that team won."})


# Create Pydantic models from JSON schemas using EasyLLM
DefaultModel = EasyLLM.generate_pydantic_model_from_json_schema("Default", DEFAULT_RESPONSE)
ResourceChangeModel = EasyLLM.generate_pydantic_model_from_json_schema("ResourceChange", RESOURCE_CHANGE_SCHEMA)
ActionResponseModel = EasyLLM.generate_pydantic_model_from_json_schema("ActionResponse", ACTION_RESPONSE_JSON_SCHEMA)
TurnModel = EasyLLM.generate_pydantic_model_from_json_schema("Turn", TURN_JSON_SCHEMA)
RoundModel = EasyLLM.generate_pydantic_model_from_json_schema("Round", ROUND_JSON_SCHEMA)
SystemUseModel = EasyLLM.generate_pydantic_model_from_json_schema("SystemUse", REQUIRES_SYSTEM_USE)
AgentReqsModel = EasyLLM.generate_pydantic_model_from_json_schema("AgentRequirements", AGENT_REQUIREMENT_SCHEMA)
MultipleAgentsModel = EasyLLM.generate_pydantic_model_from_json_schema("MultipleAgentRequirements", LIST_OF_AGENTS_SCHEMA)
WinModel = EasyLLM.generate_pydantic_model_from_json_schema("Winning", WIN_SCHEMA)