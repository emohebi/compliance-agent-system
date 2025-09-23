"""Script to manage local knowledge base."""

import json
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from tools.local_knowledge_base import LocalKnowledgeBase
from config.settings import settings

def load_sample_data(kb: LocalKnowledgeBase, sample_file: str):
    """Load sample data into knowledge base."""
    print(f"Loading sample data from: {sample_file}")
    
    try:
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
        
        documents = sample_data.get('documents', [])
        for doc in documents:
            result = kb.store(
                content=doc.get('content', ''),
                metadata=doc.get('metadata', {}),
                document_id=doc.get('id')
            )
            print(f"  - {result['message']}")
        
        print(f"\nLoaded {len(documents)} documents successfully!")
        
    except Exception as e:
        print(f"Error loading sample data: {e}")

def search_kb(kb: LocalKnowledgeBase, query: str):
    """Search the knowledge base."""
    print(f"\nSearching for: '{query}'")
    print("-" * 50)
    
    results = kb.retrieve(query, max_results=5)
    
    if results['results']:
        for i, result in enumerate(results['results'], 1):
            print(f"\nResult {i} (Score: {result['score']})")
            print(f"ID: {result['document_id']}")
            print(f"Content: {result['content'][:200]}...")
            if result['metadata']:
                print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
    else:
        print("No results found.")

def add_document(kb: LocalKnowledgeBase, content: str, metadata: str = None):
    """Add a document to the knowledge base."""
    meta_dict = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except:
            print("Warning: Invalid metadata JSON, storing without metadata")
    
    result = kb.store(content, meta_dict)
    print(f"Document added: {result['message']}")

def list_documents(kb: LocalKnowledgeBase):
    """List all documents in the knowledge base."""
    result = kb.list_documents()
    print(f"\nTotal documents: {result['total']}")
    print("-" * 50)
    
    for doc in result['documents']:
        print(f"\nID: {doc['id']}")
        print(f"Created: {doc['created_at']}")
        print(f"Preview: {doc['content_preview']}")
        if doc['metadata']:
            print(f"Metadata: {json.dumps(doc['metadata'], indent=2)}")

def show_stats(kb: LocalKnowledgeBase):
    """Show knowledge base statistics."""
    stats = kb.get_statistics()
    print("\nKnowledge Base Statistics")
    print("-" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Manage local knowledge base")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Load sample data
    load_parser = subparsers.add_parser('load-sample', help='Load sample data')
    load_parser.add_argument('--file', default='data/sample_knowledge_base.json', 
                            help='Sample data file')
    
    # Search
    search_parser = subparsers.add_parser('search', help='Search knowledge base')
    search_parser.add_argument('query', help='Search query')
    
    # Add document
    add_parser = subparsers.add_parser('add', help='Add document')
    add_parser.add_argument('content', help='Document content')
    add_parser.add_argument('--metadata', help='JSON metadata')
    
    # List documents
    list_parser = subparsers.add_parser('list', help='List documents')
    
    # Show statistics
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    
    # Initialize knowledge base
    kb = LocalKnowledgeBase(settings.LOCAL_KB_FILE)
    args.command = 'load-sample'
    args.file = str(Path(__file__).parent.parent / 'data/sample_knowledge_base.json')
    if args.command == 'load-sample':
        load_sample_data(kb, args.file)
    elif args.command == 'search':
        search_kb(kb, args.query)
    elif args.command == 'add':
        add_document(kb, args.content, args.metadata)
    elif args.command == 'list':
        list_documents(kb)
    elif args.command == 'stats':
        show_stats(kb)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()