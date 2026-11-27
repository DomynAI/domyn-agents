"""The fraud engine: publishes its alerts to the ``fraud.alerts`` Kafka topic.

The alerts are the made-up ones in ``services.py``. Give alert ids to publish only those:

    uv run --with aiokafka python alerts.py A-1001 A-1004

``--if-empty`` publishes only when the topic has no alert yet: an alert published twice is
two turns of its case. The console's "Send the five sample alerts" does the same.

See README.md to run it.
"""

import asyncio
import json
import os
import sys

from services import ALERTS

KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
ALERTS_TOPIC = "fraud.alerts"


async def published(producer) -> int:
    """How many alerts the topic holds."""
    from aiokafka import AIOKafkaConsumer
    from aiokafka.structs import TopicPartition

    partitions = await producer.partitions_for(ALERTS_TOPIC)
    consumer = AIOKafkaConsumer(bootstrap_servers=KAFKA)
    await consumer.start()
    try:
        ends = await consumer.end_offsets([TopicPartition(ALERTS_TOPIC, p) for p in partitions])
        return sum(ends.values())
    finally:
        await consumer.stop()


async def main(alert_ids: list[str], if_empty: bool = False, verbose: bool = True) -> int:
    """Publish the alerts; return how many were published."""
    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA)
    await producer.start()
    try:
        if if_empty and (count := await published(producer)):
            if verbose:
                print(f"{ALERTS_TOPIC} already holds {count} alerts: nothing published.")
            return 0
        for alert_id in alert_ids:
            alert = {"alert_id": alert_id, **ALERTS[alert_id]}
            del alert["status"]  # the services' state, not part of the alert
            await producer.send_and_wait(
                ALERTS_TOPIC, json.dumps(alert).encode(), key=alert_id.encode()
            )
            if verbose:
                print(f"{alert_id} {alert['type']}: {alert['summary']}")
        return len(alert_ids)
    finally:
        await producer.stop()


if __name__ == "__main__":
    args = sys.argv[1:]
    if_empty = "--if-empty" in args
    ids = [a for a in args if a != "--if-empty"]
    asyncio.run(main(ids or list(ALERTS), if_empty=if_empty))
