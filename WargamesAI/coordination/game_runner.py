class GameRunner:
    """
    Runs the game rounds and handles the flow of the game.
    """

    def __init__(self, game, umpire):
        """
        Initializes the GameRunner.

        Args:
            game: The game instance.
            umpire: The umpire instance.
        """
        self._game = game
        self._umpire = umpire
        self._current_round_index = 0

    def perform_round(self):
        """
        Performs the next round in the game.

        Returns:
            A dictionary with the results of the round, or False if no more rounds.
        """
        rounds = self._game._rounds

        if self._current_round_index >= len(rounds):
            return False

        round_results = {}
        current_round = rounds[self._current_round_index]

        for turn in current_round:
            response = self._umpire.engage_turn(turn)
            activity = turn["ACTIVITY"]
            round_results[activity] = response

        self._current_round_index += 1
        return round_results

    def run_all_rounds(self):
        """
        Runs all remaining rounds in the game.

        Returns:
            A dictionary with the results of all rounds.
        """
        results = {}

        rounds = self._game._rounds

        if self._current_round_index >= len(rounds):
            return False

        for round_index in range(self._current_round_index, len(rounds)):
            current_round = rounds[round_index]
            round_results = {}

            print(current_round)

            for turn_index, turn in enumerate(current_round):
                team = turn.get("TEAM", "Unknown Team")
                player = turn.get("PLAYER", "Unknown Player")
                activity = turn.get("ACTIVITY", "No Activity")
                print(
                    f"Round {round_index + 1}, Turn {turn_index + 1}: {team} - {player}: {activity}"
                )

                response = self._umpire.engage_turn(turn)
                round_results[turn_index] = response
                print(f"Response: {response}")

            results[round_index] = round_results
            self._current_round_index += 1

        return results
