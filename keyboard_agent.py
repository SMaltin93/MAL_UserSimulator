import json
import logging
import pprint
import re
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class KeyboardAgent:
    """
    Interactive attacker: each simulator tick it prints the attack-step and waits for the operator
    to type the numeric node-id to execute.  Press ENTER to skip.
    """

    name = "Keyboard Agent"         

    def __init__(self, agent_config: dict[str, Any]) -> None:
        self.attack_graph = agent_config.pop("attack_graph")
        self.logs: list[dict] = []

 
    def compute_action_from_dict(
        self, observation: dict[str, Any], mask: tuple
    ) -> tuple[int, Optional[int]]:

        actionable = list(map(int, np.flatnonzero(mask[1])))
        if not actionable:
            logger.debug("%s: no actionable steps this tick", self.name)
            return 0, None

        pretty = [
            f"{nid} → {self.attack_graph.nodes[nid].asset.name}"
            f".{self.attack_graph.nodes[nid].name}"
            for nid in actionable
        ]
        print("\n=== KeyboardAttacker ===")
        print("Available node-ids:")
        print(" • " + "\n • ".join(pretty))

        choice = input("Enter node-id to execute (ENTER to skip): ").strip()
        if not choice:
            return 0, None

        try:
            node_id = int(choice)
        except ValueError:
            print("! Invalid number")
            return 0, None

        if node_id not in actionable:
            print("! Node-id not actionable this tick")
            return 0, None

        self._collect_logs(observation, node_id)
        return 1, node_id  


    def _collect_logs(self, observation, node_id):
        for _, detector in self.attack_graph.nodes[node_id].detectors.items():
            attack_step = self.attack_graph.nodes[node_id]
            log = {
                "timestamp": observation["timestamp"],
                "_detector": detector.name,
                "asset": str(attack_step.asset.name),
                "attack_step": attack_step.name,
                "agent": self.__class__.__name__,
            }

            # /this needs to mark the log with "unknown" if it is 2-detectors in the same asset, but it is not necessary
            for label, lgasset in detector.context.items():
                matching_assets = [
                    step.asset
                    for step in self.attack_graph.attackers[0].reached_attack_steps
                    if step.asset.type in [subasset.name for subasset in lgasset.sub_assets]
                ]
                log[label] = str(matching_assets[-1].name) if matching_assets else "unknown"

            self.logs.append(log)
            logger.info('Detector triggered on %s', attack_step.full_name)
            logger.info(pprint.pformat(log))


    def terminate(self):
        with open("logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)
        self.logs.clear()
