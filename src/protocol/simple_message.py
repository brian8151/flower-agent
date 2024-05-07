
def send_message(queue, message):
    """Simulate sending a message by placing it in a queue."""
    queue.append(message)  # Send message by appending to the list

def receive_message(queue):
    """Simulate receiving a message by reading from a queue."""
    if queue:
        return queue.pop(0)  # Receive message by popping from the list
    return None