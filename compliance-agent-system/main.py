#!/usr/bin/env python3
"""
Main entry point for the Compliance Agent System.
Provides an interactive interface to interact with all system features.
"""

import sys
import os
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback
from colorama import init, Fore, Back, Style
import time

# Initialize colorama for colored output
init(autoreset=True)

# Import configuration and logging setup
from config.settings import settings
from config.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Import agents and tools
try:
    from agents.knowledge_base_agent import KnowledgeBaseAgent
    from agents.compliance_agent import ComplianceAgent
    from agents.orchestrator_agent import OrchestratorAgent
    from tools.knowledge_base_tools import kb_tools
    from tools.compliance_tools import compliance_tools
    from tools.local_knowledge_base import LocalKnowledgeBase
except ImportError as e:
    print(f"{Fore.RED}Error importing modules: {e}")
    print(f"Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class ComplianceAgentCLI:
    """Interactive CLI for the Compliance Agent System."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.kb_agent = None
        self.compliance_agent = None
        self.orchestrator = None
        self.local_kb = None
        self.session_history = []
        self.is_initialized = False
        
    def initialize_system(self):
        """Initialize all agents and tools."""
        print(f"{Fore.CYAN}Initializing Compliance Agent System...")
        
        try:
            # Check configuration
            kb_mode = settings.get_kb_mode()
            print(f"{Fore.GREEN}✓ Knowledge Base Mode: {kb_mode.upper()}")
            
            if settings.USE_LOCAL_KNOWLEDGE_BASE:
                print(f"{Fore.GREEN}✓ Local KB File: {settings.LOCAL_KB_FILE}")
                self.local_kb = LocalKnowledgeBase(settings.LOCAL_KB_FILE)
                
                # Check if KB has documents
                stats = self.local_kb.get_statistics()
                print(f"{Fore.GREEN}✓ Documents in KB: {stats['total_documents']}")
                
                if stats['total_documents'] == 0:
                    print(f"{Fore.YELLOW}⚠ No documents in knowledge base.")
                    print(f"{Fore.YELLOW}  Run: python examples/manage_local_kb.py load-sample")
            else:
                print(f"{Fore.GREEN}✓ AWS Bedrock KB ID: {settings.BEDROCK_KNOWLEDGE_BASE_ID}")
                if not settings.BEDROCK_KNOWLEDGE_BASE_ID:
                    print(f"{Fore.YELLOW}⚠ AWS KB ID not configured")
            
            # Initialize agents
            print(f"{Fore.CYAN}Initializing agents...")
            self.kb_agent = KnowledgeBaseAgent()
            self.compliance_agent = ComplianceAgent()
            self.orchestrator = OrchestratorAgent()
            
            self.is_initialized = True
            print(f"{Fore.GREEN}✅ System initialized successfully!\n")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Initialization failed: {e}")
            logger.error(f"Initialization error: {e}")
            traceback.print_exc()
            return False
        
        return True
    
    def print_header(self):
        """Print application header."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}{Style.BRIGHT}  Compliance Agent System - Interactive Console")
        print(f"{Fore.CYAN}{'='*70}")
        print(f"{Fore.WHITE}  Mode: {Fore.GREEN}{settings.get_kb_mode().upper()}")
        if settings.USE_LOCAL_KNOWLEDGE_BASE:
            print(f"{Fore.WHITE}  KB File: {Fore.GREEN}{settings.LOCAL_KB_FILE}")
        else:
            print(f"{Fore.WHITE}  AWS Region: {Fore.GREEN}{settings.AWS_REGION}")
        print(f"{Fore.CYAN}{'='*70}\n")
    
    def print_menu(self):
        """Print main menu options."""
        print(f"\n{Fore.YELLOW}Main Menu:")
        print(f"{Fore.WHITE}  1. {Fore.GREEN}Search Knowledge Base")
        print(f"{Fore.WHITE}  2. {Fore.GREEN}Check Document Compliance")
        print(f"{Fore.WHITE}  3. {Fore.GREEN}Run Compliance Workflow")
        print(f"{Fore.WHITE}  4. {Fore.GREEN}Add Document to KB {Fore.YELLOW}(Local mode only)")
        print(f"{Fore.WHITE}  5. {Fore.GREEN}View KB Statistics")
        print(f"{Fore.WHITE}  6. {Fore.GREEN}Interactive Chat with Agent")
        print(f"{Fore.WHITE}  7. {Fore.GREEN}Batch Compliance Check")
        print(f"{Fore.WHITE}  8. {Fore.GREEN}View Session History")
        print(f"{Fore.WHITE}  9. {Fore.GREEN}Settings & Configuration")
        print(f"{Fore.WHITE}  0. {Fore.RED}Exit")
        print()
    
    def search_knowledge_base(self):
        """Search the knowledge base interactively."""
        print(f"\n{Fore.CYAN}=== Search Knowledge Base ==={Style.RESET_ALL}")
        
        query = input(f"{Fore.YELLOW}Enter search query: {Style.RESET_ALL}").strip()
        if not query:
            print(f"{Fore.RED}Query cannot be empty.")
            return
        
        print(f"{Fore.CYAN}Searching for: '{query}'...")
        
        try:
            # Use KB tools to search
            results = kb_tools.retrieve_from_knowledge_base(
                query=query,
                max_results=5,
                min_score=0.1
            )
            
            if results.get('success') and results.get('results'):
                print(f"\n{Fore.GREEN}Found {len(results['results'])} results:\n")
                
                for i, result in enumerate(results['results'], 1):
                    print(f"{Fore.YELLOW}Result {i} (Score: {result['score']:.3f})")
                    print(f"{Fore.WHITE}Content: {result['content'][:300]}...")
                    
                    if result.get('metadata'):
                        print(f"{Fore.CYAN}Metadata: {json.dumps(result['metadata'], indent=2)}")
                    print("-" * 50)
            else:
                print(f"{Fore.YELLOW}No results found for '{query}'")
            
            # Add to session history
            self.session_history.append({
                'action': 'search',
                'query': query,
                'results': len(results.get('results', [])),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"{Fore.RED}Search failed: {e}")
            logger.error(f"Search error: {e}")
    
    def check_document_compliance(self):
        """Check a document for compliance violations."""
        print(f"\n{Fore.CYAN}=== Check Document Compliance ==={Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}Enter document content (type 'END' on a new line when done):")
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        
        document_content = '\n'.join(lines)
        
        if not document_content.strip():
            print(f"{Fore.RED}Document cannot be empty.")
            return
        
        print(f"{Fore.CYAN}Checking compliance...")
        
        try:
            # Use compliance agent
            result = self.compliance_agent.check_document(
                document_content=document_content,
                document_metadata={'source': 'manual_input'}
            )
            
            # Display results
            print(f"\n{Fore.YELLOW}=== Compliance Check Results ===")
            print(f"{Fore.WHITE}Is Compliant: ", end='')
            if result.is_compliant:
                print(f"{Fore.GREEN}✓ YES")
            else:
                print(f"{Fore.RED}✗ NO")
            
            print(f"{Fore.WHITE}Compliance Score: {Fore.YELLOW}{result.score:.2f}/1.00")
            
            if result.violations:
                print(f"\n{Fore.RED}Violations Found ({len(result.violations)}):")
                for v in result.violations:
                    print(f"  • {v.get('rule_name')}: {v.get('description')}")
            
            if result.warnings:
                print(f"\n{Fore.YELLOW}Warnings ({len(result.warnings)}):")
                for w in result.warnings:
                    print(f"  • {w.get('rule_name')}: {w.get('description')}")
            
            # Get remediation suggestions
            if not result.is_compliant:
                print(f"\n{Fore.CYAN}Getting remediation suggestions...")
                suggestions = self.compliance_agent.suggest_remediation(result.violations)
                print(f"{Fore.GREEN}Remediation Suggestions:")
                print(suggestions[:500])
            
            # Add to session history
            self.session_history.append({
                'action': 'compliance_check',
                'is_compliant': result.is_compliant,
                'score': result.score,
                'violations': len(result.violations),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"{Fore.RED}Compliance check failed: {e}")
            logger.error(f"Compliance check error: {e}")
    
    def run_compliance_workflow(self):
        """Run a complete compliance workflow."""
        print(f"\n{Fore.CYAN}=== Run Compliance Workflow ==={Style.RESET_ALL}")
        
        query = input(f"{Fore.YELLOW}Enter search query for documents: {Style.RESET_ALL}").strip()
        if not query:
            print(f"{Fore.RED}Query cannot be empty.")
            return
        
        check_compliance = input(f"{Fore.YELLOW}Check compliance? (y/n) [y]: {Style.RESET_ALL}").strip().lower()
        check_compliance = check_compliance != 'n'
        
        print(f"{Fore.CYAN}Processing workflow...")
        
        try:
            # Use orchestrator
            results = self.orchestrator.process_compliance_workflow(
                query=query,
                check_compliance=check_compliance
            )
            
            print(f"\n{Fore.GREEN}Workflow completed successfully!")
            print(f"{Fore.YELLOW}Steps completed: {len(results.get('steps', []))}")
            
            for i, step in enumerate(results.get('steps', []), 1):
                print(f"\n{Fore.CYAN}Step {i}: {step['step']}")
                print(f"{Fore.WHITE}Status: {Fore.GREEN}{step['status']}")
                if 'results' in step:
                    print(f"{Fore.WHITE}Results: {str(step['results'])[:200]}...")
            
            if 'summary' in results:
                print(f"\n{Fore.YELLOW}=== Workflow Summary ===")
                print(f"{Fore.WHITE}{results['summary'][:500]}")
            
            # Add to session history
            self.session_history.append({
                'action': 'workflow',
                'query': query,
                'steps': len(results.get('steps', [])),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"{Fore.RED}Workflow failed: {e}")
            logger.error(f"Workflow error: {e}")
    
    def add_document_to_kb(self):
        """Add a document to the local knowledge base."""
        if not settings.USE_LOCAL_KNOWLEDGE_BASE:
            print(f"{Fore.YELLOW}This feature is only available in local KB mode.")
            return
        
        print(f"\n{Fore.CYAN}=== Add Document to Knowledge Base ==={Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}Enter document content (type 'END' on a new line when done):")
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        
        content = '\n'.join(lines)
        
        if not content.strip():
            print(f"{Fore.RED}Document cannot be empty.")
            return
        
        # Get metadata
        print(f"{Fore.YELLOW}Enter metadata (optional, JSON format) or press Enter to skip:")
        metadata_str = input().strip()
        metadata = {}
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
            except:
                print(f"{Fore.YELLOW}Invalid JSON, proceeding without metadata.")
        
        try:
            # Store document
            result = kb_tools.store_in_knowledge_base(
                content=content,
                metadata=metadata
            )
            
            print(f"{Fore.GREEN}✓ Document stored successfully!")
            print(f"{Fore.WHITE}Document ID: {result.get('document_id')}")
            
            # Add to session history
            self.session_history.append({
                'action': 'add_document',
                'document_id': result.get('document_id'),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"{Fore.RED}Failed to store document: {e}")
            logger.error(f"Store document error: {e}")
    
    def view_kb_statistics(self):
        """View knowledge base statistics."""
        print(f"\n{Fore.CYAN}=== Knowledge Base Statistics ==={Style.RESET_ALL}")
        
        try:
            stats = kb_tools.get_knowledge_base_stats()
            
            print(f"\n{Fore.YELLOW}Knowledge Base Information:")
            for key, value in stats.items():
                if key != 'metadata_fields':
                    print(f"{Fore.WHITE}  {key}: {Fore.GREEN}{value}")
            
            if 'metadata_fields' in stats and stats['metadata_fields']:
                print(f"{Fore.WHITE}  metadata_fields: {Fore.GREEN}{', '.join(stats['metadata_fields'])}")
            
        except Exception as e:
            print(f"{Fore.RED}Failed to get statistics: {e}")
            logger.error(f"Statistics error: {e}")
    
    def interactive_chat(self):
        """Start an interactive chat session with the orchestrator."""
        print(f"\n{Fore.CYAN}=== Interactive Chat with Agent ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Type 'exit' to return to main menu")
        print(f"{Fore.YELLOW}Type 'help' for available commands\n")
        
        while True:
            try:
                user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    print(f"{Fore.YELLOW}Available commands:")
                    print(f"  - Ask any question about compliance")
                    print(f"  - Request document searches")
                    print(f"  - Ask for compliance checks")
                    print(f"  - Request reports and analysis")
                    print(f"  - Type 'exit' to return to menu")
                    continue
                elif not user_input:
                    continue
                
                print(f"{Fore.CYAN}Agent: {Style.RESET_ALL}Thinking...")
                
                # Process with orchestrator
                response = self.orchestrator.interactive_session(user_input)
                
                print(f"{Fore.CYAN}Agent: {Style.RESET_ALL}{response}")
                
                # Add to session history
                self.session_history.append({
                    'action': 'chat',
                    'user_input': user_input[:50] + '...' if len(user_input) > 50 else user_input,
                    'timestamp': datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Chat interrupted. Type 'exit' to return to menu.")
            except Exception as e:
                print(f"{Fore.RED}Chat error: {e}")
                logger.error(f"Chat error: {e}")
    
    def batch_compliance_check(self):
        """Perform batch compliance checking."""
        print(f"\n{Fore.CYAN}=== Batch Compliance Check ==={Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}Enter search queries (one per line, type 'END' when done):")
        queries = []
        while True:
            query = input().strip()
            if query == 'END':
                break
            if query:
                queries.append(query)
        
        if not queries:
            print(f"{Fore.RED}No queries provided.")
            return
        
        print(f"{Fore.CYAN}Processing {len(queries)} queries...")
        
        try:
            # Process batch
            results = self.orchestrator.batch_compliance_check(queries)
            
            if 'results' in results:
                batch_results = results['results']
                print(f"\n{Fore.GREEN}Batch Results:")
                print(f"{Fore.WHITE}Total documents: {batch_results.get('total_documents', 0)}")
                print(f"{Fore.GREEN}Compliant: {batch_results.get('compliant_documents', 0)}")
                print(f"{Fore.RED}Non-compliant: {batch_results.get('non_compliant_documents', 0)}")
                print(f"{Fore.YELLOW}Compliance rate: {batch_results.get('compliance_rate', 0)*100:.1f}%")
            
            if 'report' in results:
                print(f"\n{Fore.YELLOW}=== Compliance Report ===")
                print(results['report'][:1000])
                
                # Option to save report
                save = input(f"\n{Fore.YELLOW}Save full report to file? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w') as f:
                        f.write(results['report'])
                    print(f"{Fore.GREEN}Report saved to: {filename}")
            
            # Add to session history
            self.session_history.append({
                'action': 'batch_check',
                'queries_count': len(queries),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"{Fore.RED}Batch check failed: {e}")
            logger.error(f"Batch check error: {e}")
    
    def view_session_history(self):
        """View the session history."""
        print(f"\n{Fore.CYAN}=== Session History ==={Style.RESET_ALL}")
        
        if not self.session_history:
            print(f"{Fore.YELLOW}No actions in this session yet.")
            return
        
        for i, entry in enumerate(self.session_history, 1):
            timestamp = entry.get('timestamp', 'N/A')
            action = entry.get('action', 'unknown')
            
            print(f"\n{Fore.YELLOW}{i}. [{timestamp}]")
            print(f"{Fore.WHITE}   Action: {Fore.GREEN}{action}")
            
            # Show action-specific details
            if action == 'search':
                print(f"{Fore.WHITE}   Query: {entry.get('query', 'N/A')}")
                print(f"{Fore.WHITE}   Results: {entry.get('results', 0)}")
            elif action == 'compliance_check':
                print(f"{Fore.WHITE}   Compliant: {entry.get('is_compliant', 'N/A')}")
                print(f"{Fore.WHITE}   Score: {entry.get('score', 'N/A'):.2f}")
                print(f"{Fore.WHITE}   Violations: {entry.get('violations', 0)}")
            elif action == 'workflow':
                print(f"{Fore.WHITE}   Query: {entry.get('query', 'N/A')}")
                print(f"{Fore.WHITE}   Steps: {entry.get('steps', 0)}")
            elif action == 'batch_check':
                print(f"{Fore.WHITE}   Queries: {entry.get('queries_count', 0)}")
    
    def view_settings(self):
        """View and modify settings."""
        print(f"\n{Fore.CYAN}=== Settings & Configuration ==={Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Current Configuration:")
        print(f"{Fore.WHITE}  KB Mode: {Fore.GREEN}{settings.get_kb_mode().upper()}")
        print(f"{Fore.WHITE}  Local KB File: {Fore.GREEN}{settings.LOCAL_KB_FILE}")
        print(f"{Fore.WHITE}  AWS Region: {Fore.GREEN}{settings.AWS_REGION}")
        print(f"{Fore.WHITE}  Bedrock Model: {Fore.GREEN}{settings.BEDROCK_MODEL_ID}")
        print(f"{Fore.WHITE}  Temperature: {Fore.GREEN}{settings.AGENT_TEMPERATURE}")
        print(f"{Fore.WHITE}  Max Tokens: {Fore.GREEN}{settings.AGENT_MAX_TOKENS}")
        print(f"{Fore.WHITE}  Compliance Threshold: {Fore.GREEN}{settings.COMPLIANCE_THRESHOLD}")
        print(f"{Fore.WHITE}  Log Level: {Fore.GREEN}{settings.LOG_LEVEL}")
        
        print(f"\n{Fore.YELLOW}Note: To change settings, edit the .env file and restart the application.")
    
    def run(self):
        """Main application loop."""
        # Print header
        self.print_header()
        
        # Initialize system
        if not self.initialize_system():
            print(f"{Fore.RED}Failed to initialize system. Exiting.")
            return
        
        # Main loop
        while True:
            try:
                
                self.interactive_chat()
                
                # self.print_menu()
                
                # choice = input(f"{Fore.YELLOW}Enter your choice (0-9): {Style.RESET_ALL}").strip()
                
                # if choice == '0':
                #     print(f"\n{Fore.YELLOW}Thank you for using Compliance Agent System!")
                #     print(f"{Fore.CYAN}Goodbye! 👋\n")
                #     break
                # elif choice == '1':
                #     self.search_knowledge_base()
                # elif choice == '2':
                #     self.check_document_compliance()
                # elif choice == '3':
                #     self.run_compliance_workflow()
                # elif choice == '4':
                #     self.add_document_to_kb()
                # elif choice == '5':
                #     self.view_kb_statistics()
                # elif choice == '6':
                #     self.interactive_chat()
                # elif choice == '7':
                #     self.batch_compliance_check()
                # elif choice == '8':
                #     self.view_session_history()
                # elif choice == '9':
                #     self.view_settings()
                # else:
                #     print(f"{Fore.RED}Invalid choice. Please enter a number between 0-9.")
                
                # # Pause before showing menu again
                # if choice in '123456789':
                #     input(f"\n{Fore.YELLOW}Press Enter to continue...")
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use '0' to exit properly.")
            except Exception as e:
                print(f"{Fore.RED}An error occurred: {e}")
                logger.error(f"Main loop error: {e}")
                traceback.print_exc()


def check_environment():
    """Check if the environment is properly configured."""
    print(f"{Fore.CYAN}Checking environment configuration...")
    
    issues = []
    
    # Check .env file
    if not (Path(__file__).parent / '.env').is_file():
        issues.append("No .env file found. Copy .env.example to .env and configure it.")
    
    # Check critical settings
    if not settings.USE_LOCAL_KNOWLEDGE_BASE:
        if not settings.AWS_ACCESS_KEY_ID:
            issues.append("AWS_ACCESS_KEY_ID not configured (required for AWS mode)")
        if not settings.BEDROCK_KNOWLEDGE_BASE_ID:
            issues.append("BEDROCK_KNOWLEDGE_BASE_ID not configured (required for AWS mode)")
    else:
        # Check local KB file
        kb_file = Path(settings.LOCAL_KB_FILE)
        if not kb_file.exists():
            print(f"{Fore.YELLOW}Local KB file will be created: {settings.LOCAL_KB_FILE}")
    
    if issues:
        print(f"{Fore.RED}Configuration Issues Found:")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\n{Fore.YELLOW}Please fix these issues before continuing.")
        return False
    
    print(f"{Fore.GREEN}✓ Environment configuration looks good!\n")
    return True


def main():
    """Main entry point for the application."""
    print(f"{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{Style.BRIGHT}  Welcome to Compliance Agent System")
    print(f"{Fore.CYAN}{'='*70}\n")
    
    # Check environment
    if not check_environment():
        print(f"\n{Fore.YELLOW}Setup Instructions:")
        print(f"1. Copy .env.example to .env")
        print(f"2. Set USE_LOCAL_KNOWLEDGE_BASE=true for local mode")
        print(f"3. Or configure AWS credentials for AWS mode")
        print(f"4. Run: python examples/manage_local_kb.py load-sample")
        print(f"5. Then run this script again\n")
        sys.exit(1)
    
    # Create and run CLI
    try:
        cli = ComplianceAgentCLI()
        cli.run()
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Cleanup
    print(f"{Fore.CYAN}Session ended. Log file: {settings.LOG_FILE}")
    sys.exit(0)


if __name__ == "__main__":
    main()
