""" Ethreum Event Publisher """

import requests
import json
from src.util import log

logger = log.init_logger()


class HttpUtils:
    """ Class for HTTP message event Publisher """

    @staticmethod
    def call_get(url):
        """ Method to send HTTP GET request """
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.get(url, headers=headers)
            logger.info("HTTP GET request successful. Response: %s", response.text)
            return response
        except Exception as err:
            logger.error("Error sending HTTP GET request: %s", err, exc_info=True)
            return None

    @staticmethod
    def call_post(url, json_data):
        """ Method to send HTTP POST request """
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, data=json.dumps(json_data), headers=headers)
            logger.info("HTTP POST request successful. Response: %s", response.text)
            return response
        except Exception as err:
            logger.error("Error sending HTTP POST request: %s", err, exc_info=True)
            return None
