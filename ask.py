from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import AnalysisAgent
from agents.critic_agent import CriticAgent
from agents.orchestrator import Orchestrator
from core.llm_client import OpenAISQLClient, OpenAINarrativeClient, AnthropicCriticClient

orchestrator = Orchestrator(
    retrieval_agent=RetrievalAgent(llm_client=OpenAISQLClient()),
    analysis_agent=AnalysisAgent(llm_client=OpenAINarrativeClient()),
    critic_agent=CriticAgent(llm_client=AnthropicCriticClient()),
)

report = orchestrator.answer("Why did Home & Garden margin drop in Q2 2025?")
print(report.to_markdown())