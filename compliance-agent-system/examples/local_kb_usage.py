"""Examples using local knowledge base."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.compliance_agent import ComplianceAgent
from agents.orchestrator_agent import OrchestratorAgent

def example_1_local_kb_setup():
    """Example 1: Setup and test local knowledge base."""
    print("\n=== Example 1: Local Knowledge Base Setup ===\n")
    
    # Ensure we're using local KB
    if not settings.USE_LOCAL_KNOWLEDGE_BASE:
        print("Note: Set USE_LOCAL_KNOWLEDGE_BASE=true in .env to use local KB")
        return
    
    from tools.knowledge_base_tools import kb_tools
    
    # Store a document
    result = kb_tools.store_in_knowledge_base(
        content="This is a test compliance document. It contains information about GDPR compliance and data protection policies.",
        metadata={"type": "test", "category": "compliance"}
    )
    print(f"Store result: {result}")
    
    # Retrieve documents
    search_result = kb_tools.retrieve_from_knowledge_base(
        query="GDPR compliance",
        max_results=5
    )
    print(f"\nSearch results: {search_result}")
    
    # Get statistics
    stats = kb_tools.get_knowledge_base_stats()
    print(f"\nKB Statistics: {stats}")

def example_2_local_kb_with_agent():
    """Example 2: Use agent with local knowledge base."""
    print("\n=== Example 2: Agent with Local KB ===\n")
    
    kb_agent = KnowledgeBaseAgent()
    
    # Search for documents
    results = kb_agent.retrieve_documents(
        query="data protection encryption",
        max_results=3
    )
    
    print(f"Agent retrieved documents:")
    print(results['response'][:500])

def example_3_compliance_check_local():
    """Example 3: Compliance check with local KB."""
    print("\n=== Example 3: Compliance Check with Local KB ===\n")
    
    orchestrator = OrchestratorAgent()
    
    # Process compliance workflow with local KB
    results = orchestrator.process_compliance_workflow(
        query="PCI compliance credit card",
        check_compliance=True
    )
    
    print(f"Workflow completed with {len(results['steps'])} steps")
    print(f"Summary: {results.get('summary', 'N/A')[:500]}")

def example_4_compare_modes():
    """Example 4: Compare local vs AWS modes."""
    print("\n=== Example 4: Knowledge Base Mode Comparison ===\n")
    
    print(f"Current mode: {settings.get_kb_mode()}")
    print(f"Local KB file: {settings.LOCAL_KB_FILE}")
    
    if settings.USE_LOCAL_KNOWLEDGE_BASE:
        print("\nUsing LOCAL knowledge base:")
        print("✅ No AWS costs")
        print("✅ Works offline")
        print("✅ Fast for small datasets")
        print("✅ Full control over data")
        print("❌ Limited to TF-IDF search")
        print("❌ No vector embeddings")
    else:
        print("\nUsing AWS Bedrock knowledge base:")
        print("✅ Semantic search with embeddings")
        print("✅ Scales to millions of documents")
        print("✅ Managed infrastructure")
        print("❌ Requires AWS account and costs")
        print("❌ Needs internet connection")
        print(f"KB ID: {settings.BEDROCK_KNOWLEDGE_BASE_ID}")

def main():
    """Run all local KB examples."""
    print("=" * 60)
    print("Local Knowledge Base Examples")
    print("=" * 60)
    
    # Check configuration
    if settings.USE_LOCAL_KNOWLEDGE_BASE:
        print(f"\n✅ Local KB mode is ENABLED")
        print(f"KB File: {settings.LOCAL_KB_FILE}")
    else:
        print(f"\n⚠️ Local KB mode is DISABLED")
        print("Set USE_LOCAL_KNOWLEDGE_BASE=true in .env to enable")
    
    try:
        example_1_local_kb_setup()
        example_2_local_kb_with_agent()
        example_3_compliance_check_local()
        example_4_compare_modes()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()