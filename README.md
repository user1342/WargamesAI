<p align="center">
    <img width=100% src="assets/logo.png">
  </a>
</p>
<p align="center"> 🤖 Professional Wargaming LLM Toolbox🎲 </p>

WargamesAI is a toolkit designed to help you quickly create and run, test, and play wargames using Large Language Model agents. 

# 🎲 What Can WargamesAI Do for You?
- **Rapid Prototyping:** Quickly generate scenarios, rules, and agents to test out ideas.
- **Scalable Games:** Run large-scale wargames with dozens or even hundreds of AI agents, enabling you to explore scenarios that would be impractical with human players.
- **AI-Driven Play:** Let AI agents in silo or alongside human players interact with the Umpire to facilitate the wargame. 

WargamesAI uses Large Language Models (LLMs) to simulate reasoning and decision-making for your agents. These models generate responses and actions based on the scenarios you create, providing realistic and dynamic gameplay. WargamesAI key features, include:

- LLM facilitates Umpiring.
- LLM or Human players.
- Automatic generation of games, scenarios, rules, rounds, and players using LLMs.
- LLM-guided game summarisation.
- Dice and Card integration.
- Extraction of LLM agent personalities from text or PDFs.

# 🧠 LLM Reasoning
LLMs reason by predicting the most probable next word in a sequence based on the context provided. They use transformer architectures with attention mechanisms to capture relationships between words and phrases across long text sequences. This allows them to generate responses that follow logical and grammatical structures similar to human language use. While their outputs can mimic human reasoning patterns—such as drawing inferences, answering questions, or providing explanations—they do so without genuine comprehension or awareness. Their "reasoning" is a result of pattern recognition and statistical associations formed during training on large datasets, rather than conscious thought processes. 

In the case of wargaming this means that LLMs can provide In the case of wargaming, this means that LLMs can provide valuable simulations and strategic analyses by generating plausible scenarios, potential outcomes, and tactical suggestions based on patterns learned from extensive military and historical data. However, while they can mimic human reasoning in crafting strategies and forecasting developments, they lack true understanding and cannot account for unpredictable human factors or novel situations without prior data representation. Therefore, their use in wargaming can augment human expertise but should not replace the nuanced judgment of experienced strategists.

# 🧊 Scenario: Operation Icebreaker
In Operation Icebreaker, WargamesAI utilises the ```Agent``` class to represent each faction, with LLM decision-making guided by their deployment directives, beliefs, and objectives. The ```Game``` class manages the overall structure, including rounds and teams, while the ```Umpire``` class adjudicates actions.

```python
# Setting up the scenario
icebreaker_game = Game(
    rounds=icebreaker_rounds,
    use_dice=True,
    game_rules_text="A strategic game focused on controlling key resources and positions in the Arctic.",
    teams={"Polar Alliance": polar_alliance_agents, "Arctic Consortium": arctic_consortium_agents}
)

# Creating agents
polar_agent = Agent(
    game=icebreaker_game,
    deployment_directive="Secure Arctic resources and maintain strategic dominance.",
    factions=["Polar Alliance"],
    beliefs=["Economic superiority", "Territorial integrity"],
    disposition="Aggressive and determined",
    empathy="Low",
    exercise_objectives=["Control key Arctic points", "Deter other factions"],
    strategic_objectives=["Ensure long-term dominance in the Arctic"],
    bio_folder="./bios"
)

consortium_agent = Agent(
    game=icebreaker_game,
    deployment_directive="Expand influence and gain access to Arctic resources.",
    factions=["Arctic Consortium"],
    beliefs=["Expansion", "Economic growth"],
    disposition="Cautious but ambitious",
    empathy="Medium",
    exercise_objectives=["Negotiate key alliances", "Avoid direct conflict"],
    strategic_objectives=["Secure a foothold in the Arctic"],
    bio_folder="./bios"
)

# Running the game
umpire = Umpire(game=icebreaker_game)
game_runner = GameRunner(game=icebreaker_game, umpire=umpire)
results = game_runner.run_all_rounds()

print(umpire.produce_summary())
print(umpire.deduce_winner())

```

# ⚙️ Setup
Tomato required Nvidia CUDA. Follow the steps below:
- Ensure your Nvidia drivers are up to date: https://www.nvidia.com/en-us/geforce/drivers/
- Install the appropriate dependancies from here: https://pytorch.org/get-started/locally/
- Validate CUDA is installed correctly by running the following and being returned a prompt ```python -c "import torch; print(torch.rand(2,3).cuda())"```
  
Install the dependencies using:

```bash
git clone https://github.com/user1342/WargamesAI.git
cd WargamesAI
pip install -r requirements.txt
pip install -e .
```
or
```bash
pip install git+https://github.com/user1342/WargamesAI.git
```

# 📚 Examples

## Creating an LLM player from a PDF
```python
# Example: Creating an Agent with a PDF Bio
pdf_agent = Agent(
    game=icebreaker_game,
    pdf_bio="path/to/bio.pdf",
    bio_folder="./bios"
)
```

## Creating a Game from a rules PDF
```python
# Example: Creating a Game with PDF Rules
pdf_rules_game = Game(
    rounds=icebreaker_rounds,
    game_rules_pdf="path/to/rules.pdf",
    teams={"Polar Alliance": polar_alliance_agents, "Arctic Consortium": arctic_consortium_agents}
)
Changing the Model Used for Agents and Umpire
WargamesAI allows you to customize the AI models used by your agents and umpire. You might want to change the model to fit different scenarios or performance needs.

```

## Setting custom models for the player character agents
```python
# Example: Using a Different Model for an Agent
custom_model_agent = Agent(
    game=icebreaker_game,
    deployment_directive="Lead negotiations to secure strategic resources.",
    factions=["Diplomatic Corps"],
    beliefs=["Peaceful resolution", "Economic benefit"],
    disposition="Diplomatic",
    empathy="High",
    exercise_objectives=["Achieve a treaty"],
    strategic_objectives=["Secure resources without conflict"],
    model_name="unsloth/mistral-7b-instruct-v0.3",  # Custom model
    bio_folder="./bios"
)
```

## Setting custom models for Umpire role
```python
# Example: Using a Different Model for the Umpire
custom_model_umpire = Umpire(
    game=icebreaker_game,
    model_name="unsloth/Llama-2-7b-bnb-4bit",  # Custom model
    max_tokens=800
)
```

## Rolling Dice [Experimental]
```python
# Example: Rolling Dice
dice_result = umpire.roll_dice(highest=6, times=2)  # Rolling two 6-sided dice
print(f"Dice roll result: {dice_result}")
```

## Drawing Cards [Experimental]
```python
# Example: Drawing Cards
cards = ["Sabotage", "Reinforcements", "Diplomatic Pressure"]
drawn_cards = umpire.pick_card(number=2)  # Drawing two cards from the deck
print(f"Drawn cards: {drawn_cards}")
Using PDF Bios and PDF Rules
WargamesAI allows you to import bios and rules from PDFs, making it easier to manage detailed information about agents and game settings.
```

# 🙏 Contributions
WargamesAI is an open-source project and welcomes contributions from the community. If you would like to contribute to WargamesAI, please follow these guidelines:

- Fork the repository to your own GitHub account.
- Create a new branch with a descriptive name for your contribution.
- Make your changes and test them thoroughly.
- Submit a pull request to the main repository, including a detailed description of your changes and any relevant documentation.
- Wait for feedback from the maintainers and address any comments or suggestions (if any).
- Once your changes have been reviewed and approved, they will be merged into the main repository.

# ⚖️ Code of Conduct
WargamesAI follows the Contributor Covenant Code of Conduct. Please make sure to review and adhere to this code of conduct when contributing to Tomato.

# 🐛 Bug Reports and Feature Requests
If you encounter a bug or have a suggestion for a new feature, please open an issue in the GitHub repository. Please provide as much detail as possible, including steps to reproduce the issue or a clear description of the proposed feature. Your feedback is valuable and will help improve WargamesAI for everyone.

# 📜 License
 GPL-3.0 license
