import re
from typing import Any, Dict, List, Tuple, Type, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, RootModel, create_model
import json
import os
import random

# Suppress unnecessary warnings
hf_logging.set_verbosity_error()

UNSLOTH_MODELS = ["unsloth/Llama-3.1-Storm-8B-bnb-4bit",
                  "unsloth/mistral-7b-instruct-v0.3",
                  "unsloth/llama-3-8b-Instruct-bnb-4bit",
                  "unsloth/gemma-2-9b-it-bnb-4bit",
                  "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
                  "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
                  "unsloth/Phi-3.5-mini-instruct-bnb-4bit"]

class EasyLLM:
    """
    A simple class for interacting with a pretrained language model to generate dialogue responses.
    """

    def __init__(
        self,
        max_new_tokens: int = 200,
        model_name: str = None,
    ) -> None:
        """
        Initializes the EasyLLM class with a specified model and token generation limit.

        Args:
            max_new_tokens (int): Maximum number of new tokens to generate in a response.
            model_name (str): Name of the pretrained language model to use.
        """
        self.max_new_tokens = max_new_tokens
        
        if model_name is None:
            model_name = random.choice(UNSLOTH_MODELS)
            print(f"No model chosen, model {model_name} selected.")
        
        self.model_name = model_name
        self.dialogue: List[dict] = []

        self._device: str = "cuda"
        self.model = None
        self.tokenizer = None

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the pretrained language model and tokenizer only when needed.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded language model and tokenizer.
        """
        if self.model is None or self.tokenizer is None:
            is_4bit = '4bit' in self.model_name.lower()
            is_8bit = '8bit' in self.model_name.lower()

            if is_4bit or is_8bit:
                # Use BitsAndBytesConfig for quantized models
                if is_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.bfloat16,
                    )

                max_memory = {
                    0: "13GiB",  # Adjust this value based on your GPU's available memory
                    "cpu": "30GiB"  # Adjust based on your system's RAM
                }

                # Use device_map with max_memory to control layer placement
                device_map = "auto"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder="offload",  # Folder to offload weights if necessary
                )
            else:
                # For non-quantized models
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

            # Ensure pad_token_id is set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        return self.model, self.tokenizer

    def _unload_model(self) -> None:
        """
        Unloads the model from GPU memory by deleting the model and tokenizer.
        """
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

    def _generate_dialogue_response(self, messages: List[dict]) -> str:
        """
        Generates a response from the language model based on the input messages.

        Args:
            messages (List[dict]): List of input messages.

        Returns:
            str: Generated response from the language model.
        """
        # Load model and tokenizer if not already loaded
        self._load_model()
        
        chat_template = getattr(self.tokenizer, 'chat_template', None)
        if (chat_template):
            # Use the chat template to prepare input
            input_data = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            # Check if input_data is a dictionary or tensor
            if isinstance(input_data, dict):
                input_ids = input_data["input_ids"].to(self._device)
                attention_mask = input_data.get("attention_mask", torch.ones_like(input_ids)).to(self._device)
            elif isinstance(input_data, torch.Tensor):
                # input_data is a tensor
                input_ids = input_data.to(self._device)
                attention_mask = torch.ones_like(input_ids).to(self._device)
            else:
                raise TypeError(f"Unexpected type for input_data: {type(input_data)}")
        else:
            # Manually format the prompt without assuming roles
            prompt = self.format_messages(messages)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            attention_mask = torch.ones_like(input_ids).to(self._device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Extract only the newly generated tokens
        generated_tokens = generated_ids[:, input_ids.shape[-1]:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        # Unload model after generation to free up GPU memory
        self._unload_model()

        return decoded.strip()

    def reset_dialogue(self) -> None:
        """
        Resets the dialogue history, clearing all previous messages.
        """
        self.dialogue = []

    def ask_question(self, question: str, reset_dialogue: bool = False) -> str:
        """
        Generates a response for the given question using the loaded model.

        Args:
            question (str): The question or prompt provided by the user.
            reset_dialogue (bool): Whether to reset the dialogue history after generating a response.

        Returns:
            str: Generated response to the question.
        """
        if reset_dialogue:
            self.reset_dialogue()

        # Load model and tokenizer if not already loaded
        self._load_model()

        # Extract roles from the chat template
        chat_template = getattr(self.tokenizer, 'chat_template', None)
        roles = self.extract_roles_from_template(chat_template) if chat_template else []

        # Determine the roles for the messages
        message_roles = self.get_message_roles(roles)

        # Add the user's question with the appropriate role
        if message_roles['user']:
            self.dialogue.append({"role": message_roles['user'], "content": question})
        else:
            self.dialogue.append({"content": question})

        # Prepare the messages for the model
        messages_for_model = self.dialogue.copy()

        # Generate the response
        result = self._generate_dialogue_response(messages_for_model)

        # Add the model's response to the dialogue history
        if message_roles['assistant']:
            self.dialogue.append({"role": message_roles['assistant'], "content": result})
        else:
            self.dialogue.append({"content": result})

        # Optionally reset the dialogue
        if reset_dialogue:
            self.reset_dialogue()

        return result

    def extract_roles_from_template(self, chat_template: str) -> List[str]:
        """
        Extracts roles used in the chat template.

        Args:
            chat_template (str): The chat template string.

        Returns:
            List[str]: A list of roles extracted from the template.
        """
        # Use regex to find roles in the chat template
        roles = re.findall(r'message\["role"\]\s*==\s*"([^"]+)"', chat_template)
        return list(set(roles))

    def get_message_roles(self, roles: List[str]) -> Dict[str, str]:
        """
        Determines the roles to use for the messages.

        Args:
            roles (List[str]): List of roles extracted from the chat template.

        Returns:
            Dict[str, str]: A dictionary with keys 'user' and 'assistant' mapping to the appropriate roles.
        """
        if not roles:
            # If no roles are found, default to 'user' and 'assistant'
            return {'user': 'user', 'assistant': 'assistant'}

        # Heuristic to assign roles
        user_role = None
        assistant_role = None

        for role in roles:
            role_lower = role.lower()
            if 'user' in role_lower and not user_role:
                user_role = role
            elif 'assistant' in role_lower and not assistant_role:
                assistant_role = role
            elif 'system' in role_lower and not assistant_role:
                assistant_role = role  # Assuming 'system' acts as assistant if no 'assistant' role

        if not user_role:
            # Assign any role that's not assistant as user
            for role in roles:
                if role != assistant_role:
                    user_role = role
                    break

        if not user_role and not assistant_role:
            # Assign default roles
            user_role = 'user'
            assistant_role = 'assistant'

        if not user_role:
            user_role = 'user'
        if not assistant_role:
            assistant_role = 'assistant'

        return {'user': user_role, 'assistant': assistant_role}

    def format_messages(self, messages: List[dict]) -> str:
        """
        Formats messages into a prompt string.

        Args:
            messages (List[dict]): List of message dictionaries.

        Returns:
            str: The formatted prompt.
        """
        formatted = ""
        for message in messages:
            content = message["content"]
            role = message.get("role")
            if role:
                formatted += f"{role}: {content}\n"
            else:
                formatted += f"{content}\n"
        return formatted.strip()

    def parse_llm_json(self, llm_response):
        """
        Parses a JSON response from the language model.

        Args:
            llm_response (str): The raw response from the language model.

        Returns:
            dict: The parsed JSON data.
        """
        # Remove backslashes if they are escape characters
        llm_response = llm_response.replace("\\", "")

        # Extract content between triple backticks
        match = re.search(r'```.*?({.*?}).*?```', llm_response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to decode JSON: {e}")
        else:
            raise Exception(f"Failed to parse JSON from: '{llm_response}'")

    def generate_json_prompt(self, schema: Type[BaseModel], query: str) -> str:
        """
        Generates a JSON prompt based on a given schema and query.

        Args:
            schema (Type[BaseModel]): The Pydantic schema to structure the output.
            query (str): The query for which the JSON response is to be generated.

        Returns:
            str: The JSON-formatted prompt.
        """
        # Ensure schema is a class (Type[BaseModel])
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("The schema argument must be a Pydantic model class.")

        # Initialize the JSON output parser with your schema
        parser = JsonOutputParser(pydantic_object=schema)

        # Create a prompt template with the format instructions injected
        prompt = PromptTemplate(
            template="Answer the following query in JSON format according to the provided schema:\n{format_instructions}\n\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Generate the response using the model and parser
        response = prompt.format(query=query)

        return response

    @staticmethod
    def parse_field(field_name: str, field_value: Any) -> tuple:
        """
        Parses a field name and its value from the JSON schema to determine the Pydantic field type and field settings.

        Args:
            field_name (str): The name of the field in the schema.
            field_value (Any): The value or type of the field in the schema.

        Returns:
            tuple: A tuple containing the type and Field settings for the Pydantic model.
        """
        if isinstance(field_value, str):
            return (str, Field(description=field_value))
        elif isinstance(field_value, int):
            return (int, Field(description=field_value))
        elif isinstance(field_value, bool):
            return (bool, Field(description=field_value))
        elif isinstance(field_value, list):
            if field_value and isinstance(field_value[0], dict):
                return (List[EasyLLM.generate_pydantic_model_from_json_schema(f"{field_name}_item", field_value[0])], ...)
            return (List[Any], Field(description="List of values"))
        elif isinstance(field_value, dict):
            return (EasyLLM.generate_pydantic_model_from_json_schema(field_name.capitalize(), field_value), ...)
        else:
            return (Any, Field(description="Unknown field type"))

    @staticmethod
    def generate_pydantic_model_from_json_schema(
        schema_name: str, json_schema: Union[str, Dict[str, Any], List[Any]]
    ) -> Type[BaseModel]:
        """
        Generates a Pydantic model from a JSON schema.

        Args:
            schema_name (str): The name of the Pydantic model to create.
            json_schema (Union[str, Dict[str, Any], List[Any]]): The JSON schema as a string, dictionary, or list.

        Returns:
            Type[BaseModel]: The generated Pydantic model class.
        """
        if isinstance(json_schema, str):
            json_schema = json.loads(json_schema)

        if isinstance(json_schema, dict):
            fields = {
                field_name: EasyLLM.parse_field(field_name, field_value)
                for field_name, field_value in json_schema.items()
            }
            return create_model(schema_name, **fields)

        elif isinstance(json_schema, list):
            if json_schema and isinstance(json_schema[0], dict):
                # Handle list of dictionaries by creating a model for the elements
                return RootModel[List[EasyLLM.generate_pydantic_model_from_json_schema(f"{schema_name}_item", json_schema[0])]]
            else:
                return RootModel[List[Any]]

        else:
            raise ValueError("The provided JSON schema must be a dictionary, list, or valid JSON string.")
