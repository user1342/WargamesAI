import os
from WargamesAI.utils.easyLLM import EasyLLM
from WargamesAI.utils.easyRAG import EasyRAG
from WargamesAI.utils import json_schemas, pdf_utils

class Agent:
    """
    Represents a player character (agent) in the game.
    """

    def __init__(
        self,
        game,
        pdf_bio=None,
        deployment_directive=None,
        factions=None,
        beliefs=None,
        disposition=None,
        empathy=None,
        exercise_objectives=None,
        strategic_objectives=None,
        notes=None,
        max_tokens=1000,
        is_human=False,
        model_name=None,
        bio_folder=".",
    ):
        """
        Initializes an Agent instance.

        Args:
            game: The game instance.
            pdf_bio (str): Path to the PDF bio of the agent.
            deployment_directive (str): The deployment directive for the agent.
            factions (list): List of factions the agent belongs to.
            beliefs (list): List of beliefs of the agent.
            disposition (str): The disposition of the agent.
            empathy (str): The empathy level of the agent.
            exercise_objectives (list): List of exercise objectives.
            strategic_objectives (list): List of strategic objectives.
            notes (dict): Additional notes.
            max_tokens (int): Maximum tokens for LLM responses.
            model_name (str): Name of the language model to use.
            bio_folder (str): Folder to store the bio PDFs.
        """
        if pdf_bio is None and deployment_directive is None:
            raise ValueError("Both 'deployment_directive' and 'pdf_bio' cannot be None!")

        self.game = game
        self.llm = EasyLLM(max_new_tokens=max_tokens, model_name=model_name)
        self.rag = EasyRAG()
        self.action_history = []

        self._pdf_bio = pdf_bio
        self._deployment_directive = deployment_directive

        # Initialize parameters, generating them from the PDF if not provided
        self._factions = factions
        self._beliefs = beliefs
        self._disposition = disposition
        self._empathy = empathy
        self._exercise_objectives = exercise_objectives
        self._strategic_objectives = strategic_objectives
        self._notes = notes if notes is not None else {}

        self._is_human = is_human

        if pdf_bio:
            # Generate parameters from the PDF if not provided
            self._generate_params_from_pdf()

        if not self._deployment_directive:
            self._deployment_directive = self._generate_param_from_pdf(
                "What is the overall objective of this individual?", as_list=False
            )

        # Generate the user bio
        self._bio_text = self._generate_user_bio()

        # Hash the bio text to create a unique name
        self._name = pdf_utils.hash_string(self._bio_text)

        # If pdf_bio was not provided, create a PDF bio
        if not pdf_bio:

            if not os.path.exists(bio_folder):
                os.mkdir(bio_folder)
                
            self._pdf_bio = os.path.join(bio_folder, f"{self._name}.pdf")
            pdf_utils.write_pdf(self._bio_text, self._pdf_bio)

        # Initialize the agent in the LLM
        generated_prompt = self.llm.generate_json_prompt(
            json_schemas.DefaultModel,
            f"You are a player in the game with the following rules. Be ready to play the game. Enjoy!\n\n"
            f"Game Rules:\n'{self.game._game_rules_text}'.\n\nYour player character bio:\n'{self._bio_text}'",
        )
        self.llm.ask_question(generated_prompt)

    def _generate_params_from_pdf(self):
        """
        Generates missing parameters from the provided PDF bio using the RAG model.
        """

        pdf_data = pdf_utils.read_pdf(self._pdf_bio)
        self._disposition = self._disposition or self.llm.ask_question(
            self.llm.generate_json_prompt(
                json_schemas.DefaultModel, f"What is this individual's disposition? Answer concisely. {pdf_data}"
            )
        )

        print(self._disposition)
        exit()

        self._empathy = self._empathy or self.llm.ask_question(
            self.llm.generate_json_prompt(
                json_schemas.DefaultModel, f"What is the empathy level of this individual? Answer concisely. {pdf_data}"
            )
        )["RESPONSE"]

        self._exercise_objectives = self._exercise_objectives or self.llm.ask_question(
            self.llm.generate_json_prompt(
                json_schemas.DefaultModel, f"What are this individual's exercise objectives? Answer concisely. {pdf_data}"
            )
        )["RESPONSE"]

        self._strategic_objectives = self._strategic_objectives or self.llm.ask_question(
            self.llm.generate_json_prompt(
                json_schemas.DefaultModel, f"What are this individual's strategic objectives? Answer concisely. {pdf_data}"
            )
        )["RESPONSE"]

    def _generate_param_from_pdf(self, question, as_list=False):
        """
        Generates a parameter by querying the RAG model with the PDF bio.

        Args:
            question (str): The question to ask.
            as_list (bool): Whether the expected response is a list.

        Returns:
            The response from the RAG model.
        """
        if not self._pdf_bio:
            return [] if as_list else ""

        response = self.rag.ask_question_with_pdf(question, self._pdf_bio)

        if as_list:
            return [item.strip() for item in response.split(",")] if response else []
        return response

    def _generate_user_bio(self):
        """
        Generates the user bio text based on the agent's attributes.

        Returns:
            str: The user bio text.
        """
        bio_parts = []

        if self._deployment_directive:
            bio_parts.append(f"Deployment Directive: {self._deployment_directive}")

        if self._factions:
            bio_parts.append(f"Factions: {', '.join(self._factions)}")

        if self._beliefs:
            bio_parts.append(f"Beliefs: {', '.join(self._beliefs)}")

        if self._disposition:
            bio_parts.append(f"Disposition: {self._disposition}")

        if self._empathy:
            bio_parts.append(f"Empathy: {self._empathy}")

        if self._exercise_objectives:
            bio_parts.append(f"Exercise Objectives: {', '.join(self._exercise_objectives)}")

        if self._strategic_objectives:
            bio_parts.append(f"Strategic Objectives: {', '.join(self._strategic_objectives)}")

        if self._notes:
            notes_str = "Notes:\n" + "\n".join(f"  - {key}: {value}" for key, value in self._notes.items())
            bio_parts.append(notes_str)

        return "\n".join(bio_parts)

    def request_action(self, scenario, attempts=5):
        """
        Requests an action from the agent based on a scenario.

        Args:
            scenario (str): The scenario to present to the agent.
            attempts (int): Maximum number of attempts to generate an in-character action.

        Returns:
            The agent's action response if valid, else False.
        """
        original_prompt = f"Scenario: {scenario}"
        for attempt in range(attempts):
            response = self.llm.ask_question(
                self.llm.generate_json_prompt(json_schemas.ActionResponseModel, original_prompt)
            )

            is_in_character = self.rag.ask_question_with_pdf(
                f"Would this user perform the following action? Action: {response}", self._pdf_bio
            )

            if is_in_character:
                self.action_history.append(response)
                return response
            else:
                original_prompt = f"Your last response was deemed out of character. Try again. {scenario}"

        return False  # Failed to generate a valid action

    @property
    def name(self):
        """Returns the agent's name."""
        return self._name
