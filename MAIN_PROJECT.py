from flask import Flask, render_template, request, jsonify
import requests
import os
import time
import redis
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import BaseTool
from typing import Type, List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Configuration
class Config:
    def __init__(self):
        # Try Groq first, fallback to OpenAI
        try:
            self.llm = ChatOpenAI(
                openai_api_base="https://api.groq.com/openai/v1",
                openai_api_key="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",
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.7,
                max_tokens=1000
            )
            # Test the connection
            self.llm.invoke("Test")
            logger.info("Connected to Groq API")
        except Exception as e:
            logger.warning(f"Groq connection failed, falling back to OpenAI: {str(e)}")
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )

        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.cache_ttl = 3600
        self.max_concurrent_requests = 100
        self.safety_threshold = 0.85
        self.debate_rounds = 3
        self.max_history_length = 10


config = Config()

# Initialize Redis
try:
    redis_client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        decode_responses=True
    )
    redis_client.ping()
except redis.ConnectionError as e:
    logger.error(f"Redis connection error: {e}")
    redis_client = None


# Initialize models
class ModelManager:
    def __init__(self):
        self.llm = config.llm
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.rate_limit_last_call = 0

    def get_embeddings(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    def generate_text(self, prompt: str) -> str:
        try:
            # Rate limiting
            time_since_last = time.time() - self.rate_limit_last_call
            if time_since_last < 1.0:  # 1 second between calls
                time.sleep(1.0 - time_since_last)

            response = self.llm.invoke(prompt)
            self.rate_limit_last_call = time.time()
            return response.content
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise Exception("Failed to generate response")


model_manager = ModelManager()


# Agentic Model Manager
class AgenticModelManager:
    def __init__(self):
        self.llm = config.llm
        self.tools = self._setup_tools()
        self.agent_executors = {}

    def _setup_tools(self) -> List[Tool]:
        """Define tools that agents can use"""
        debate_tool = Tool(
            name="DebateAnalyzer",
            func=self.analyze_debate_context,
            description="Useful for analyzing current debate context and positions"
        )

        research_tool = Tool(
            name="FactChecker",
            func=self.check_facts,
            description="Useful for verifying facts and claims during debate"
        )

        reasoning_tool = Tool(
            name="LogicalReasoner",
            func=self.logical_reasoning,
            description="Useful for performing logical analysis of arguments"
        )

        ethics_tool = Tool(
            name="EthicsEvaluator",
            func=self.evaluate_ethics,
            description="Useful for evaluating ethical implications of arguments"
        )

        return [debate_tool, research_tool, reasoning_tool, ethics_tool]

    def analyze_debate_context(self, query: str) -> str:
        """Analyze the current debate context"""
        prompt = f"""
        Analyze this debate context and summarize key positions:
        {query}

        Provide:
        1. Key arguments from each side
        2. Areas of agreement/disagreement
        3. Suggestions for moving forward
        """
        return model_manager.generate_text(prompt)

    def check_facts(self, query: str) -> str:
        """Fact-checking tool"""
        prompt = f"""
        Verify the factual accuracy of these claims:
        {query}

        For each claim:
        1. State if it's factually correct
        2. Provide evidence/sources
        3. Note any uncertainties
        """
        return model_manager.generate_text(prompt)

    def logical_reasoning(self, query: str) -> str:
        """Logical reasoning tool"""
        prompt = f"""
        Perform logical analysis of these arguments:
        {query}

        Identify:
        1. Logical fallacies if any
        2. Strength of reasoning
        3. Potential counter-arguments
        """
        return model_manager.generate_text(prompt)

    def evaluate_ethics(self, query: str) -> str:
        """Ethics evaluation tool"""
        prompt = f"""
        Evaluate the ethical implications of:
        {query}

        Consider:
        1. Potential harms/benefits
        2. Rights affected
        3. Justice/fairness considerations
        4. Virtue ethics perspective
`        """
        return model_manager.generate_text(prompt)

    def get_agent_executor(self, agent_type: str) -> AgentExecutor:
        """Get or create an agent executor for a specific type"""
        if agent_type not in self.agent_executors:
            self.agent_executors[agent_type] = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5,
                early_stopping_method="generate",
                agent_kwargs={
                    "prefix": self._get_agent_prefix(agent_type),
                    "suffix": self._get_agent_suffix(agent_type)
                }
            )
        return self.agent_executors[agent_type]

    def _get_agent_prefix(self, agent_type: str) -> str:
        """Get the agent-specific prefix"""
        prefixes = {
            "ethicist": """You are an AI ethics expert with deep knowledge of ethical frameworks and AI policy. 
            Your role is to analyze ethical implications of arguments and proposals. 
            Focus on identifying potential harms, benefits, and ethical trade-offs.""",

            "technologist": """You are a technology expert specializing in AI systems development. 
            Focus on technical feasibility, implementation challenges, and practical considerations. 
            Evaluate arguments based on technical merit and real-world applicability.""",

            "philosopher": """You are a moral philosopher with expertise in ethical theories. 
            Examine arguments through the lenses of deontology, utilitarianism, virtue ethics, and other philosophical frameworks. 
            Focus on underlying principles and values.""",

            "scientist": """You are a research scientist with expertise in empirical methods. 
            Focus on evidence-based arguments, data quality, and scientific validity. 
            Identify gaps in evidence and suggest needed research.""",

            "policy": """You are a policy expert with experience in technology governance. 
            Consider regulatory frameworks, policy implications, and governance structures. 
            Evaluate arguments based on practical policy implementation."""
        }
        return prefixes.get(agent_type, "You are a knowledgeable expert.")

    def _get_agent_suffix(self, agent_type: str) -> str:
        """Get the agent-specific suffix"""
        return """Begin! And remember to:
        - Think step by step before answering
        - Consider multiple perspectives
        - Justify your reasoning with evidence
        - Cite sources when possible
        - Maintain professional and respectful tone
        - Acknowledge limitations of your knowledge
        - Clearly distinguish facts from opinions"""


agentic_model_manager = AgenticModelManager()


# Agent Memory Implementation
class AgentMemory:
    def __init__(self, max_memory_items: int = 1000):
        self.memory = []
        self.max_memory_items = max_memory_items

    def add_memory(self, content: str, importance: float = 1.0) -> None:
        embedding = model_manager.get_embeddings(content)
        self.memory.append({
            'content': content,
            'embedding': embedding,
            'timestamp': datetime.now().isoformat(),
            'importance': min(max(importance, 0.0), 1.0)
        })
        if len(self.memory) > self.max_memory_items:
            self.memory = sorted(
                self.memory,
                key=lambda x: (-x['importance'], x['timestamp'])
            )[:self.max_memory_items]

    def retrieve_relevant(self, query: str, top_k: int = 3) -> List[str]:
        if not self.memory:
            return []
        query_embed = model_manager.get_embeddings(query)
        embeddings = np.array([item['embedding'] for item in self.memory])
        similarities = np.dot(embeddings, query_embed)
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        return [self.memory[i]['content'] for i in top_indices[np.argsort(-similarities[top_indices])]]

        # Cognitive Agent Implementation


class CognitiveAgent:
    def __init__(self, role: str, expertise: str, agent_type: str):
        self.role = role
        self.expertise = expertise
        self.agent_type = agent_type
        self.memory = AgentMemory()
        self.conversation_history: List[Tuple[str, str]] = []
        self.personality_traits = self._initialize_personality()
        self.agent_executor = agentic_model_manager.get_agent_executor(agent_type)

    def _initialize_personality(self) -> Dict[str, float]:
        return {
            'openness': np.random.uniform(0.5, 1.0),
            'conscientiousness': np.random.uniform(0.3, 0.9),
            'skepticism': np.random.uniform(0.2, 0.8),
            'creativity': np.random.uniform(0.4, 1.0),
            'confidence': np.random.uniform(0.3, 0.9),
            'diplomacy': np.random.uniform(0.5, 1.0)
        }

    def generate_response(self, prompt: str, context: Optional[List[str]] = None) -> str:
        full_prompt = self._build_prompt(prompt, context or [])
        cache_key = f"agent:{self.role}:{hash(full_prompt)}"

        if redis_client:
            if cached := redis_client.get(cache_key):
                return cached

        try:
            # First try agentic approach
            response = self.agent_executor.run(input=full_prompt)

            if redis_client:
                redis_client.setex(cache_key, config.cache_ttl, response)

            self.memory.add_memory(prompt, importance=0.7)
            self.memory.add_memory(response, importance=0.8)
            self.conversation_history.append((prompt, response))
            return response
        except Exception as e:
            logger.error(f"Agentic response generation failed, falling back to direct generation: {e}")
            try:
                # Fallback to direct generation
                response = model_manager.generate_text(full_prompt)
                if redis_client:
                    redis_client.setex(cache_key, config.cache_ttl, response)
                return response
            except Exception as e2:
                logger.error(f"Direct generation also failed: {e2}")
                return "I'm unable to respond at this time due to technical difficulties."

    def _build_prompt(self, prompt: str, context: List[str]) -> str:
        memory_context = self.memory.retrieve_relevant(prompt)
        prompt_parts = [
            f"Role: {self.role} | Expertise: {self.expertise}",
            "Personality Traits:",
            *[f"- {k}: {v:.2f}" for k, v in self.personality_traits.items()],
            "\nContext:",
            *(context or ["No additional context"]),
            "\nRelevant Memories:",
            *(memory_context or ["No relevant memories"]),
            "\nConversation History:",
            *(f"{role}: {content}" for role, content in self.conversation_history[-3:]),
            "\nCurrent Query:",
            prompt,
            "\nProvide a well-reasoned response considering your role and expertise:"
        ]
        return "\n".join(prompt_parts)


# Debate Components
class BayesianConsensus:
    def __init__(self):
        self.opinions: Dict[str, List[float]] = {}
        self.points: List[str] = []

    def update(self, agent: str, response: str) -> None:
        sentiment = self._analyze_sentiment(response)
        key_points = self._extract_key_points(response)
        if agent not in self.opinions:
            self.opinions[agent] = []
        self.opinions[agent].append(sentiment)
        self.points.extend(key_points)

    def has_consensus(self, threshold: float = 0.8) -> bool:
        if not self.opinions:
            return False
        all_sentiments = [s for sentiments in self.opinions.values() for s in sentiments]
        return np.var(all_sentiments) < (1 - threshold)

    def get_consensus_score(self) -> float:
        if not self.opinions:
            return 0.0
        all_sentiments = [s for sentiments in self.opinions.values() for s in sentiments]
        return 1 - np.var(all_sentiments)

    def extract_key_points(self, top_n: int = 5) -> List[str]:
        point_counts: Dict[str, int] = {}
        for point in self.points:
            point_counts[point] = point_counts.get(point, 0) + 1
        return [p[0] for p in sorted(point_counts.items(), key=lambda x: -x[1])[:top_n]]

    def _analyze_sentiment(self, text: str) -> float:
        positive = ['agree', 'support', 'yes', 'correct', 'benefit', 'should', 'recommend']
        negative = ['disagree', 'against', 'no', 'wrong', 'harm', 'should not', 'caution']
        text_lower = text.lower()
        pos = sum(text_lower.count(word) for word in positive)
        neg = sum(text_lower.count(word) for word in negative)
        total = pos + neg
        return pos / total if total > 0 else 0.5

    def _extract_key_points(self, text: str) -> List[str]:
        sentences = [s.strip() for s in text.split('.') if 5 < len(s.split()) <= 30]
        return sentences[:3]


class DebateOrchestrator:
    def __init__(self, agents: List[CognitiveAgent]):
        self.agents = agents
        self.debate_history: List[str] = []
        self.consensus_model = BayesianConsensus()

    def conduct_debate(self, topic: str, rounds: int = 3) -> Dict:
        self.debate_history = [f"Debate Topic: {topic}"]
        for round_num in range(rounds):
            self.debate_history.append(f"\n--- Round {round_num + 1} ---")
            for agent in self.agents:
                context = self._get_context_for_agent(agent)
                prompt = self._build_agent_prompt(topic, round_num, context)
                try:
                    response = agent.generate_response(prompt, context)
                    self.debate_history.append(f"{agent.role}: {response}")
                    self.consensus_model.update(agent.role, response)
                    if self.consensus_model.has_consensus(config.safety_threshold):
                        self.debate_history.append("\nConsensus reached!")
                        return self._compile_results()
                except Exception as e:
                    logger.error(f"Agent {agent.role} failed: {e}")
                    self.debate_history.append(f"{agent.role}: [Unable to respond - {str(e)}]")
                    continue
        return self._compile_results()

    def _get_context_for_agent(self, agent: CognitiveAgent) -> List[str]:
        return [
                   entry for entry in reversed(self.debate_history[-10:])
                   if not entry.startswith(agent.role) and ':' in entry
               ][:2]

    def _build_agent_prompt(self, topic: str, round_num: int, context: List[str]) -> str:
        return (
            f"Debate Topic: {topic}\n"
            f"Round: {round_num + 1}/{config.debate_rounds}\n"
            "Recent Arguments:\n"
            f"{'\n'.join(context) if context else 'No arguments yet'}\n\n"
            "Provide your perspective, considering these points. "
            "Be concise but substantive. Support your arguments with reasoning and evidence."
        )

    def _compile_results(self) -> Dict:
        return {
            "history": self.debate_history,
            "consensus_score": round(self.consensus_model.get_consensus_score(), 2),
            "key_points": self.consensus_model.extract_key_points(),
            "participants": [agent.role for agent in self.agents],
            "personality_traits": {agent.role: agent.personality_traits for agent in self.agents}
        }


# Initialize agents with enhanced roles and agent types
debate_agents = [
    CognitiveAgent("AI Ethicist", "AI Ethics and Policy", "ethicist"),
    CognitiveAgent("Technologist", "AI Systems Development", "technologist"),
    CognitiveAgent("Philosopher", "Moral Philosophy", "philosopher"),
    CognitiveAgent("Research Scientist", "Empirical AI Research", "scientist"),
    CognitiveAgent("Policy Expert", "Technology Governance", "policy")
]
orchestrator = DebateOrchestrator(debate_agents)


# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def debate_form():
    if request.method == 'POST':
        topic = request.form.get('topic', '').strip()
        try:
            rounds_raw = int(request.form.get('rounds', 3))
            rounds = max(1, min(5, rounds_raw))
        except ValueError:
            rounds = 3

        if not topic or len(topic) < 10:
            return render_template('error.html',
                                   error_message="Topic must be at least 10 characters")

        try:
            start_time = time.time()
            results = orchestrator.conduct_debate(topic, rounds)
            processing_time = time.time() - start_time

            formatted_history = []
            for item in results['history']:
                if '--- Round' in item:
                    formatted_history.append({"type": "round", "content": item})
                elif ':' in item:
                    role, content = item.split(':', 1)
                    formatted_history.append({
                        "type": "response",
                        "role": role.strip(),
                        "content": content.strip()
                    })
                else:
                    formatted_history.append({"type": "info", "content": item})

            return render_template('results.html',
                                   topic=topic,
                                   rounds=rounds,
                                   history=formatted_history,
                                   key_points=results['key_points'],
                                   consensus_score=results['consensus_score'],
                                   processing_time=f"{processing_time:.2f}",
                                   participants=results['participants'],
                                   personalities=results['personality_traits'])
        except Exception as e:
            logger.error(f"Debate failed: {str(e)}")
            return render_template('error.html',
                                   error_message="An error occurred during the debate")

    return render_template('form.html', agents=[
        {"role": agent.role, "expertise": agent.expertise, "type": agent.agent_type}
        for agent in debate_agents
    ])


@app.route('/api/debate', methods=['POST'])
def api_debate():
    data = request.get_json()
    topic = data.get('topic', '').strip()
    try:
        rounds_raw = int(data.get('rounds', 3))
        rounds = max(1, min(5, rounds_raw))
    except ValueError:
        rounds = 3

    if not topic or len(topic) < 10:
        return jsonify({"error": "Topic must be at least 10 characters"}), 400

    try:
        results = orchestrator.conduct_debate(topic, rounds)
        return jsonify(results)
    except Exception as e:
        logger.error(f"API debate failed: {str(e)}")
        return jsonify({"error": "Debate processing failed"}), 500


# Template creation
def create_templates():
    os.makedirs('templates', exist_ok=True)

    templates = {
        'form.html': '''<!DOCTYPE html>
<html>
<head>
    <title>AI Debate System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input, textarea, select { width: 100%; padding: 8px; box-sizing: border-box; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .agent-badge { display: inline-block; margin: 0 5px 5px 0; padding: 5px 10px; background: #e0e0e0; border-radius: 3px; }
        .agent-type { font-size: 0.8em; color: #666; }
    </style>
</head>
<body>
    <h1>Start a Debate</h1>
    <form method="POST" action="/">
        <div class="form-group">
            <label for="topic">Debate Topic:</label>
            <textarea id="topic" name="topic" rows="3" required></textarea>
        </div>
        <div class="form-group">
            <label for="rounds">Number of Rounds:</label>
            <select id="rounds" name="rounds">
                <option value="1">1 Round</option>
                <option value="2">2 Rounds</option>
                <option value="3" selected>3 Rounds</option>
                <option value="4">4 Rounds</option>
                <option value="5">5 Rounds</option>
            </select>
        </div>
        <div class="form-group">
            <label>Participants (select at least 2):</label>
            <div>
                {% for agent in agents %}
                    <div class="agent-badge">
                        <input type="checkbox" id="agent-{{ loop.index }}" name="agents" value="{{ agent.type }}" checked>
                        <label for="agent-{{ loop.index }}">{{ agent.role }} <span class="agent-type">({{ agent.expertise }})</span></label>
                    </div>
                {% endfor %}
            </div>
        </div>
        <button type="submit">Start Debate</button>
    </form>
</body>
</html>''',

        'results.html': '''<!DOCTYPE html>
<html>
<head>
    <title>Debate Results</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .round { margin: 25px 0 15px; color: #2c3e50; font-weight: bold; }
        .response { margin: 15px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; border-left: 4px solid #3498db; }
        .response-role { font-weight: bold; color: #2980b9; }
        .key-points { margin: 30px 0; }
        .key-point { padding: 10px; margin: 5px 0; background: #e8f4fc; border-radius: 3px; }
        .consensus { margin: 15px 0; padding: 10px; background: #e8f8f0; border-radius: 3px; }
        .processing-info { color: #7f8c8d; font-size: 0.9em; }
        .personality-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .personality-card { padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .trait { margin: 5px 0; }
        .trait-name { font-weight: bold; }
        .trait-value { display: inline-block; width: 100%; height: 10px; background: #eee; }
        .trait-fill { height: 100%; background: #3498db; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Debate Results</h1>
        <h2>{{ topic }}</h2>
        <p class="processing-info">
            Processed in {{ processing_time }} seconds with {{ rounds }} rounds | 
            Consensus: {{ "%.0f"|format(consensus_score * 100) }}%
        </p>
    </div>

    <div class="personality-grid">
        {% for role, traits in personalities.items() %}
        <div class="personality-card">
            <h3>{{ role }}</h3>
            {% for name, value in traits.items() %}
            <div class="trait">
                <span class="trait-name">{{ name|title }}</span>
                <div class="trait-value">
                    <div class="trait-fill" style="width: {{ value * 100 }}%"></div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>

    <div class="debate-history">
        {% for item in history %}
            {% if item.type == "round" %}
                <div class="round">{{ item.content }}</div>
            {% elif item.type == "response" %}
                <div class="response">
                    <div class="response-role">{{ item.role }}:</div>
                    <div>{{ item.content }}</div>
                </div>
            {% else %}
                <p>{{ item.content }}</p>
            {% endif %}
        {% endfor %}
    </div>

    <div class="key-points">
        <h3>Key Points:</h3>
        {% for point in key_points %}
            <div class="key-point">{{ point }}</div>
        {% endfor %}
    </div>

    <a href="/">Start New Debate</a>
</body>
</html>''',

        'error.html': '''<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .error-box { padding: 20px; background: #ffebee; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Error Occurred</h1>
    <div class="error-box">
        <p>{{ error_message }}</p>
    </div>
    <a href="/">Return to Debate Form</a>
</body>
</html>'''
    }

    for filename, content in templates.items():
        path = os.path.join('templates', filename)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(content)


if __name__ == '__main__':
    create_templates()
    app.run(host='0.0.0.0', port=5000, debug=True)     # now here it should be converted to speech