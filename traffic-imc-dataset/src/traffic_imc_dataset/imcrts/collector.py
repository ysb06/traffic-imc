import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from json.decoder import JSONDecodeError

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

SERVICE_URL = "http://apis.data.go.kr/6280000/ICRoadVolStat/NodeLink_Trfc_DD"
MAX_ROW_COUNT = 5000


class IMCRTSCollector:
    def __init__(
        self,
        api_key: str,
        start_date: str = "20230101",
        end_date: str = "20260101",
    ) -> None:
        self.params = {
            "serviceKey": api_key,
            "pageNo": 1,
            "numOfRows": MAX_ROW_COUNT,
            "YMD": "20240101",
        }
        self.start_date: datetime = datetime.strptime(start_date, "%Y%m%d")
        self.end_date: datetime = datetime.strptime(end_date, "%Y%m%d")
        self.data: Optional[pd.DataFrame] = None

    def collect(
        self,
        ignore_empty: bool = False,
        req_delay: float = 0.1,
    ) -> None:
        total_size = (self.end_date - self.start_date).days + 1

        data_list: List[List[Dict[str, Any]]] = []
        current_date: datetime = self.start_date

        logger.info(
            f'Collecting IMCRTS Data from "{SERVICE_URL}" between {self.start_date} and {self.end_date}'
        )
        with tqdm(total=total_size) as bar:
            while current_date <= self.end_date:
                current_date_string = current_date.strftime("%Y%m%d")
                self.params["YMD"] = current_date_string
                bar.set_description(current_date_string, refresh=True)

                time.sleep(req_delay)

                code, data = self.get_data(self.params)
                if code == 200:
                    if data is not None:
                        data_list.extend(data)
                    else:
                        if ignore_empty:
                            logger.warning("Skipping...")
                        else:
                            logger.error("Aborted due to empty data")
                            break
                else:
                    logger.error(f"Error Code: {code}")
                    logger.error(f"Failed to Get Data at [{current_date_string}]")
                    break

                current_date += timedelta(days=1)
                bar.update(1)

        df = pd.DataFrame(data_list)
        self.data = df
        logger.info(f"Data Collecting Finished: {self.data.shape}")

    def to_pickle(self, output_path: str) -> None:
        logger.info("Creating Pickle...")
        self.data.to_pickle(output_path)

    def to_excel(self, output_dir: str, file_name: str = "imcrts_data.xlsx") -> None:
        logger.info("Creating Excel...")
        self.data.to_excel(os.path.join(output_dir, file_name))

    def get_data(
        self, params: Dict[str, Any]
    ) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
        """Request Data from Data Server
        Sends a GET request to SERVICE_URL.

        Args:
            params (Dict[str, Any]): Parameters for Request

        Returns:
            Tuple[int, Optional[List[Dict[str, Any]]]]: Result of Data Request
        """
        res = requests.get(url=SERVICE_URL, params=params)
        data: Optional[List[Dict[str, Any]]] = None
        if res.status_code == 200:
            try:
                raw = res.json()
            except JSONDecodeError:
                logger.error("JSON Decoding Failed")
                if "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in res.text:
                    logger.error("You may use not valid service key")
                return 0, []

            if raw["response"]["header"]["resultCode"] != "00":
                logger.warning(
                    f"Error Code: {raw['response']['header']['resultCode']}. You need to check the reason of error."
                )

            if "items" in raw["response"]["body"] and raw["response"]["body"]["items"]:
                data = raw["response"]["body"]["items"]

                if len(data) > MAX_ROW_COUNT:
                    message = f"Length of Data at {params['YMD']} is {len(data)} but sliced to {MAX_ROW_COUNT}"
                    logger.warning(message)
            else:
                logger.warning(f"No data at {params['YMD']}")
        else:
            logger.error(f"Request failed with status code {res.status_code}")
            logger.error(res.text)

        return (res.status_code, data)


class IMCRTSExcelConverter:
    """Convert pickle data produced by IMCRTSCollector to Excel.

    Use this helper when openpyxl is unavailable in the main collection flow.
    """

    def __init__(
        self,
        output_dir: str = "./datasets/imcrts/",
        filename: str = "imcrts_data.pkl",
        start_date: str = "20230101",
        end_date: str = "20231231",
    ) -> None:
        self.output_dir = output_dir
        self.filename = filename
        self.filepath = os.path.join(self.output_dir, self.filename)
        logger.info(f"Loading Data from {self.filepath}...")
        self.data: pd.DataFrame = pd.read_pickle(self.filepath)

        start_date_obj = datetime.strptime(start_date, "%Y%m%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y%m%d").date()

        first_date = datetime.strptime(self.data["statDate"].min(), "%Y-%m-%d").date()
        last_date = datetime.strptime(self.data["statDate"].max(), "%Y-%m-%d").date()

        # Compare only the date part
        if first_date != start_date_obj:
            raise ValueError(f"{first_date} is not match to {start_date_obj}")
        if last_date != end_date_obj:
            raise ValueError(f"{last_date} is not match to {end_date_obj}")

    def export(self, excel_file_name: str = "imcrts_data.xlsx") -> None:
        logger.info("Exporting Data to Excel...")
        self.data.to_excel(os.path.join(self.output_dir, excel_file_name))
