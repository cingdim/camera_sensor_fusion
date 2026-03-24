import json
import paho.mqtt.client as mqtt
import os

from enum import Enum
from typing import Union
from abc import ABC, abstractmethod

from ntp_facade_smr import TimeBrokerFacade
from time import ctime


# Conda facade-sdk

class InvalidClientType(Exception):

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data
        self.message = message

class AlreadyConnectedError(Exception):

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data
        self.message = message

class NotConnectedError(Exception):

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data
        self.message = message

class PublishError(Exception):

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data
        self.message = message

class Client:

    IMU = "imu"
    ROBOT = "robot"
    CAMERA = "camera"
    AI = "ai"

    def __init__(self, broker_ip: str, client_type: str, device_id: str, auto_connect: bool = True, broker_port: int = 1883, timeout: int = 60):

        # Ensure client type is valid
        self.client_type = client_type.lower()
        self._validate_client_type()

        self.config = {
            "BROKER_IP": broker_ip,
            "BROKER_PORT": broker_port,
            "BROKER_TIMEOUT": timeout,
        }

        self.device_id = device_id
        self.mqtt = None

        if auto_connect:
            self.connect()

    # Validates the client type and ensures it matches the correct type
    def _validate_client_type(self):

        ct = self.client_type

        if ct != Client.IMU and ct != Client.ROBOT and ct != Client.CAMERA and ct != Client.TWINS and ct != Client.AI:
            raise InvalidClientType(
                f"Incorret client type ({ct}). Only {Client.IMU}, {Client.ROBOT}, {Client.CAMERA}, {Client.TWINS}, and {Client.AI} are valid."
            )

    # Connects to the broker -- Call to try connection
    def connect(self):

        # Check if the connection already exists
        if self.mqtt is not None:
            raise AlreadyConnectedError(
                f"[{self.client_type} : {self.device_id}] is already connected to the broker. Disconnect it before trying to reconnect."
            )

        self.mqtt = mqtt.Client()
        rc = self.mqtt.connect(self.config["BROKER_IP"], self.config["BROKER_PORT"], self.config["BROKER_TIMEOUT"])

        # Check if the connection failed to connect
        if rc != 0:
            raise ConnectionError(f"Failed to connect, return code {rc}. Check if IP is correct.")

        self.mqtt.loop_start()
    
    # Disconnects from the broker
    def disconnect(self):

        # Check if the connection does not exist
        if self.mqtt is not None:
            raise NotConnectedError(
                f"[{self.client_type} : {self.device_id}] is not connected to the broker."
            )

        self.mqtt.loop_stop()
        self.mqtt.disconnect()
    
    # Pushes data to MQTT broker -- Works with Strings (convert json to string first)
    def publish(self, message: str) -> None:

        # Check if message is acceptable
        if not isinstance(message, str):
            raise PublishError(f"Invalid message type: {type(message).__name__}.", data=message)
        
        try:
            payload = json.dumps(payload) if isinstance(message, (dict, list)) else message

            # Public to [client type] / [device_id]
            self.mqtt.publish(f"{self.client_type}/{self.device_id}", payload)
        except Exception as e:
            raise PublishError(f"MQTT failed to publish {payload}: {e}")

    def get_time(self, ntp_port = 123):

        try:
            tbroker = TimeBrokerFacade(ntp_server_ip = self.config["BROKER_IP"])
        
            return tbroker.get_synchronized_time()

        except(ValueError, IOError) as e:
            print("error")
            print (e)


    # Safty disconnect from the broker
    def __del__(self):
        try:
            self.disconnect()
        except:
            pass
    