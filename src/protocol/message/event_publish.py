import asyncio

import nats
from src.config import get_config

async def event_publish(topic, json_message):
    """Method to publish a message to a NATS topic."""
    nats_url = get_config("nats_url")
    nc = await nats.connect(nats_url)
    await nc.publish(topic, json_message.encode("utf-8"))
    await nc.close()


def start_event_publish(topic, json_message):
    asyncio.run(event_publish(topic, json_message))
