import os
from typing import List

import logging

logger = logging.getLogger(__name__)


class IdList:
    @staticmethod
    def import_from_txt(filepath: str) -> "IdList":
        with open(filepath, "r") as file:
            raw = file.read().split(",")
        return IdList(raw)

    def __init__(self, raw: List[str]) -> None:
        self.data: List[str] = raw

    def to_txt(self, filepath: str) -> None:
        logger.info(f"Saving METR-IMC IDs to {filepath}...")
        ids_str = ",".join(str(id) for id in self.data)
        with open(filepath, "w") as file:
            file.write(ids_str)
        logger.info("Saving Complete")
