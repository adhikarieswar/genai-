# tasks.py

class Task:
    def __init__(self, description, expected_output, agent, priority="Normal"):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.priority = priority  # New attribute for task priority

    def execute(self):
        print(f"Executing task: {self.description}")
        print(f"Expected output: {self.expected_output}")
        print(f"Using agent: {self.agent}")
        print(f"Priority: {self.priority}")

# main.py

