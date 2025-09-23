"""Examples of using MCP-enhanced agents."""

import asyncio
import json
import logging
from agents.mcp_enhanced_agent import MCPEnhancedAgent
from agents.multi_mcp_orchestrator import MultiMCPOrchestrator
from config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def example_1_mcp_enhanced_agent():
    """Example 1: Using MCP-enhanced agent."""
    print("\n=== Example 1: MCP Enhanced Agent ===\n")
    
    agent = MCPEnhancedAgent()
    
    # Process document compliance with MCP tools
    result = agent.process_document_compliance(
        query="customer data protection policies",
        check_gdpr=True
    )
    
    print("Query:", result['query'])
    print("Analysis:", result['analysis'][:500])
    print("Token Usage:", result['usage'])

def example_2_batch_pii_scanning():
    """Example 2: Batch PII scanning with MCP."""
    print("\n=== Example 2: Batch PII Scanning ===\n")
    
    agent = MCPEnhancedAgent()
    
    documents = [
        "John Doe's email is john@example.com and his SSN is 123-45-6789",
        "Contact our support at support@company.com or call 555-123-4567",
        "Payment accepted from card ending in 4532"
    ]
    
    results = agent.batch_scan_pii(documents, mask=True)
    
    for result in results:
        print(f"Document {result['document_index']}:")
        print(result['scan_results'][:300])
        print("-" * 50)

def example_3_multi_mcp_orchestration():
    """Example 3: Multi-MCP orchestration."""
    print("\n=== Example 3: Multi-MCP Orchestration ===\n")
    
    orchestrator = MultiMCPOrchestrator()
    
    # Execute full compliance audit workflow
    audit_results = orchestrator.execute_workflow(
        "full_compliance_audit",
        {
            "scope": "financial and customer data policies",
            "include_analytics": True
        }
    )
    
    print("Workflow:", audit_results['workflow'])
    print("Status:", audit_results['status'])
    print("Steps completed:", len(audit_results.get('steps', [])))
    
    if 'final_report' in audit_results:
        print("\nFinal Report Preview:")
        print(audit_results['final_report'][:500])

def example_4_specialized_agents():
    """Example 4: Creating specialized agents with specific MCP servers."""
    print("\n=== Example 4: Specialized Agents ===\n")
    
    orchestrator = MultiMCPOrchestrator()
    
    # Create compliance-focused agent
    compliance_agent = orchestrator.create_specialized_agent(
        'compliance',
        ['compliance']
    )
    
    # Create research-focused agent
    research_agent = orchestrator.create_specialized_agent(
        'research',
        ['knowledge_base', 'analytics']
    )
    
    # Use compliance agent
    compliance_response = compliance_agent(
        "Check this text for PII: John's email is john@example.com"
    )
    print("Compliance Agent Response:", compliance_response.content[:300])
    
    # Use research agent
    research_response = research_agent(
        "Find all documents about data retention policies"
    )
    print("Research Agent Response:", research_response.content[:300])

def example_5_regulatory_assessment():
    """Example 5: Regulatory assessment workflow."""
    print("\n=== Example 5: Regulatory Assessment ===\n")
    
    orchestrator = MultiMCPOrchestrator()
    
    assessment_results = orchestrator.execute_workflow(
        "regulatory_assessment",
        {
            "regulations": ["GDPR", "HIPAA", "PCI-DSS"],
            "scope": "all organizational data handling procedures"
        }
    )
    
    print("Regulations assessed:", assessment_results['regulations'])
    print("\nAssessment summaries:")
    
    for assessment in assessment_results.get('assessments', []):
        print(f"\n{assessment['regulation']}:")
        print(assessment['assessment'][:200] + "...")

async def example_6_parallel_workflows():
    """Example 6: Execute multiple workflows in parallel."""
    print("\n=== Example 6: Parallel Workflow Execution ===\n")
    
    orchestrator = MultiMCPOrchestrator()
    
    workflows = [
        {
            "name": "full_compliance_audit",
            "parameters": {"scope": "financial data"}
        },
        {
            "name": "regulatory_assessment",
            "parameters": {"regulations": ["GDPR"]}
        }
    ]
    
    results = await orchestrator.execute_parallel_workflows(workflows)
    
    for i, result in enumerate(results):
        print(f"\nWorkflow {i+1}: {result.get('workflow', 'Unknown')}")
        print(f"Status: {result.get('status', 'Unknown')}")

def example_7_executive_reporting():
    """Example 7: Generate executive report with MCP."""
    print("\n=== Example 7: Executive Reporting ===\n")
    
    agent = MCPEnhancedAgent()
    
    search_queries = [
        "data protection policies",
        "customer information handling",
        "security incident procedures",
        "compliance training materials"
    ]
    
    report = agent.generate_executive_report(search_queries)
    
    print("Executive Report Preview:")
    print(report[:1000])

def main():
    """Run all MCP examples."""
    print("=" * 60)
    print("MCP Server Integration Examples")
    print("=" * 60)
    
    try:
        # Run synchronous examples
        example_1_mcp_enhanced_agent()
        example_2_batch_pii_scanning()
        example_3_multi_mcp_orchestration()
        example_4_specialized_agents()
        example_5_regulatory_assessment()
        example_7_executive_reporting()
        
        # Run async example
        asyncio.run(example_6_parallel_workflows())
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    main()