"""Basic usage examples for the compliance agent system."""

import asyncio
import logging
from config.logging_config import setup_logging
from agents.orchestrator_agent import OrchestratorAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.compliance_agent import ComplianceAgent

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def example_1_simple_retrieval():
    """Example 1: Simple document retrieval from knowledge base."""
    print("\n=== Example 1: Simple Document Retrieval ===\n")
    
    kb_agent = KnowledgeBaseAgent()
    
    # Search for specific information
    results = kb_agent.retrieve_documents(
        query="data protection policies",
        max_results=5,
        min_score=0.5
    )
    
    print(f"Query: {results['query']}")
    print(f"Response: {results['response']}")
    print(f"Token usage: {results['usage']}")

def example_2_compliance_check():
    """Example 2: Check document compliance."""
    print("\n=== Example 2: Compliance Check ===\n")
    
    compliance_agent = ComplianceAgent()
    
    # Sample document with potential compliance issues
    sample_document = """
    Customer Information:
    Name: John Doe
    Email: john.doe@example.com
    SSN: 123-45-6789
    Credit Card: 4532-1234-5678-9012
    
    Account Details:
    Password: MySecretPass123
    API_KEY: sk-1234567890abcdef
    
    This document contains sensitive customer data that must be protected
    according to GDPR and PCI-DSS requirements.
    """
    
    results = compliance_agent.check_document(
        document_content=sample_document,
        document_metadata={'document_type': 'customer_record'}
    )
    
    print(f"Is Compliant: {results.is_compliant}")
    print(f"Compliance Score: {results.score:.2f}")
    print(f"Violations: {len(results.violations)}")
    print(f"Warnings: {len(results.warnings)}")
    
    if results.violations:
        print("\nViolations found:")
        for violation in results.violations:
            print(f"  - {violation['rule_name']}: {violation['description']}")

def example_3_orchestrated_workflow():
    """Example 3: Full orchestrated workflow."""
    print("\n=== Example 3: Orchestrated Workflow ===\n")
    
    orchestrator = OrchestratorAgent()
    
    # Process a complete workflow
    results = orchestrator.process_compliance_workflow(
        query="financial records and transactions",
        check_compliance=True
    )
    
    print(f"Workflow Query: {results['query']}")
    print(f"Steps Completed: {len(results['steps'])}")
    print(f"\nSummary:\n{results['summary']}")

def example_4_batch_processing():
    """Example 4: Batch compliance checking."""
    print("\n=== Example 4: Batch Processing ===\n")
    
    orchestrator = OrchestratorAgent()
    
    # Check multiple document queries
    queries = [
        "employee records",
        "financial statements",
        "customer data",
        "security policies"
    ]
    
    results = orchestrator.batch_compliance_check(queries)
    
    if 'report' in results:
        print("Compliance Report Generated:")
        print(results['report'][:500] + "...")  # Print first 500 chars

def example_5_interactive_session():
    """Example 5: Interactive session with orchestrator."""
    print("\n=== Example 5: Interactive Session ===\n")
    
    orchestrator = OrchestratorAgent()
    
    # Example user requests
    requests = [
        "Search for GDPR compliance documents",
        "Check if our customer data handling is compliant",
        "What are the best practices for data protection?",
        "Generate a compliance report for Q4"
    ]
    
    for request in requests:
        print(f"\nUser: {request}")
        response = orchestrator.interactive_session(request)
        print(f"Agent: {response[:300]}...")  # Print first 300 chars

async def example_6_async_operations():
    """Example 6: Asynchronous operations (if needed)."""
    print("\n=== Example 6: Async Operations ===\n")
    
    # This is a placeholder for async operations
    # Strands supports async operations for streaming
    print("Async operations can be implemented for streaming responses")

def main():
    """Run all examples."""
    print("=" * 60)
    print("Compliance Agent System - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_1_simple_retrieval()
        example_2_compliance_check()
        example_3_orchestrated_workflow()
        example_4_batch_processing()
        example_5_interactive_session()
        
        # Run async example
        asyncio.run(example_6_async_operations())
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    main()