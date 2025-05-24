# Advanced Demo: Distributed AI Agent Collaboration
# This demonstrates multiple AI agents working together across distributed nodes

from easyremote import ComputeNode, remote
import asyncio
import json
import time
from typing import Dict, List, Any

# ================================
# AI Agent Nodes
# ================================

# Research Agent Node
research_agent_node = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="research-agent",
    capabilities={"specialization": "research", "web_access": True}
)

@research_agent_node.register
async def research_topic(topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
    """Research a topic and gather comprehensive information"""
    print(f"🔍 Research Agent: Investigating '{topic}' with {depth} depth")
    
    # Simulate research process
    await asyncio.sleep(2)  # Research time
    
    # Mock research results
    research_data = {
        "topic": topic,
        "depth": depth,
        "key_findings": [
            f"Key insight 1 about {topic}",
            f"Key insight 2 about {topic}",
            f"Key insight 3 about {topic}"
        ],
        "data_sources": [
            f"Academic paper on {topic}",
            f"Industry report about {topic}",
            f"Expert analysis of {topic}"
        ],
        "confidence_score": 0.85,
        "research_time": time.time()
    }
    
    print(f"📚 Research completed: {len(research_data['key_findings'])} findings")
    return research_data

# Analysis Agent Node
analysis_agent_node = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="analysis-agent",
    capabilities={"specialization": "analysis", "gpu": True}
)

@analysis_agent_node.register(gpu_required=True)
async def analyze_data_patterns(research_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze patterns and trends in research data using AI"""
    print(f"📊 Analysis Agent: Processing data about '{research_data['topic']}'")
    
    await asyncio.sleep(3)  # GPU analysis time
    
    analysis_result = {
        "topic": research_data["topic"],
        "pattern_analysis": {
            "trend_direction": "positive",
            "growth_rate": 15.7,
            "market_sentiment": "optimistic",
            "risk_factors": ["Factor 1", "Factor 2", "Factor 3"]
        },
        "predictive_insights": {
            "6_month_forecast": "Strong growth expected",
            "key_opportunities": ["Opportunity A", "Opportunity B"],
            "challenges": ["Challenge X", "Challenge Y"]
        },
        "confidence_score": 0.88,
        "analysis_method": "AI-powered pattern recognition"
    }
    
    print(f"🎯 Analysis completed: {analysis_result['pattern_analysis']['trend_direction']} trend detected")
    return analysis_result

# Writing Agent Node
writing_agent_node = ComputeNode(
    vps_address="gateway.example.com:8080",
    node_id="writing-agent",
    capabilities={"specialization": "content_generation", "llm": "gpt-4"}
)

@writing_agent_node.register
async def generate_report(research_data: Dict[str, Any], analysis_data: Dict[str, Any], 
                         report_type: str = "executive") -> Dict[str, Any]:
    """Generate comprehensive report from research and analysis"""
    print(f"✍️ Writing Agent: Creating {report_type} report about '{research_data['topic']}'")
    
    await asyncio.sleep(4)  # Writing time
    
    # Generate report content
    executive_summary = f"""
    Executive Summary: {research_data['topic']}
    
    Based on comprehensive research and AI-powered analysis, this report presents 
    key findings about {research_data['topic']}. Our analysis indicates a 
    {analysis_data['pattern_analysis']['trend_direction']} trend with 
    {analysis_data['pattern_analysis']['growth_rate']}% growth potential.
    """
    
    report = {
        "title": f"Comprehensive Report: {research_data['topic']}",
        "report_type": report_type,
        "executive_summary": executive_summary.strip(),
        "detailed_findings": research_data['key_findings'],
        "analysis_insights": analysis_data['predictive_insights'],
        "methodology": "AI-assisted research and analysis",
        "confidence_rating": min(research_data['confidence_score'], analysis_data['confidence_score']),
        "word_count": len(executive_summary.split()),
        "generated_at": time.time()
    }
    
    print(f"📝 Report generated: {report['word_count']} words")
    return report

# ================================
# Client Side - Agent Orchestration
# ================================

@remote(node_id="research-agent")
async def research_topic(topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
    pass

@remote(node_id="analysis-agent")
async def analyze_data_patterns(research_data: Dict[str, Any]) -> Dict[str, Any]:
    pass

@remote(node_id="writing-agent")
async def generate_report(research_data: Dict[str, Any], analysis_data: Dict[str, Any], 
                         report_type: str = "executive") -> Dict[str, Any]:
    pass

class DistributedAISystem:
    """Advanced distributed AI system with multiple specialized agents"""
    
    def __init__(self):
        self.performance_metrics = {
            "projects_completed": 0,
            "average_duration": 0.0,
            "success_rate": 0.0
        }
    
    async def run_research_project(self, topic: str):
        """Run a complete research project using distributed agents"""
        print(f"🚀 Starting research project: {topic}")
        start_time = time.time()
        
        # Step 1: Research
        research_data = await research_topic(topic, "comprehensive")
        
        # Step 2: Analysis
        analysis_data = await analyze_data_patterns(research_data)
        
        # Step 3: Generate report
        report_data = await generate_report(research_data, analysis_data, "executive")
        
        total_time = time.time() - start_time
        
        result = {
            "topic": topic,
            "total_time": total_time,
            "research_confidence": research_data["confidence_score"],
            "analysis_confidence": analysis_data["confidence_score"],
            "report_word_count": report_data["word_count"],
            "agents_used": 3
        }
        
        print(f"✅ Project completed in {total_time:.2f}s")
        return result

async def run_distributed_ai_demo():
    """Run comprehensive distributed AI agents demo"""
    print("🤖 EasyRemote Distributed AI Agents Demo")
    print("=" * 55)
    
    ai_system = DistributedAISystem()
    
    # Demo: Research project
    result = await ai_system.run_research_project("Quantum Computing Trends 2024")
    print(f"📊 Project metrics: {json.dumps(result, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(run_distributed_ai_demo()) 