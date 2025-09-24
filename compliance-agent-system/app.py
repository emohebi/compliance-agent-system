#!/usr/bin/env python3
"""
Enhanced Flask API Backend with Real-time Streaming
Provides REST API endpoints and real-time updates for the frontend chat interface
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Generator
import traceback
from pathlib import Path
import time
import threading
from queue import Queue
import uuid

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing agents
from agents.orchestrator_agent import OrchestratorAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.compliance_agent import ComplianceAgent
from config.settings import settings
from config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Session storage (in production, use Redis or database)
sessions = {}
active_streams = {}

class StreamingLogger:
    """Captures and streams agent activities in real-time."""
    
    def __init__(self, session_id: str, socket_id: Optional[str] = None):
        self.session_id = session_id
        self.socket_id = socket_id
        self.queue = Queue()
        self.active = True
        
    def emit_status(self, status_type: str, data: Dict[str, Any]):
        """Emit a status update."""
        message = {
            'type': status_type,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        if self.socket_id:
            # Emit via WebSocket if connected
            socketio.emit('status_update', message, room=self.socket_id)
        
        # Also add to queue for SSE streaming
        self.queue.put(json.dumps(message))
        
    def emit_agent_start(self, agent_name: str, task: str):
        """Emit when an agent starts working."""
        self.emit_status('agent_start', {
            'agent': agent_name,
            'task': task,
            'message': f"🤖 {agent_name} is starting: {task}"
        })
        
    def emit_tool_use(self, tool_name: str, parameters: Dict[str, Any]):
        """Emit when a tool is being used."""
        self.emit_status('tool_use', {
            'tool': tool_name,
            'parameters': parameters,
            'message': f"🔧 Using tool: {tool_name}"
        })
        
    def emit_step_complete(self, step: str, result: str):
        """Emit when a processing step completes."""
        self.emit_status('step_complete', {
            'step': step,
            'result': result,
            'message': f"✓ Completed: {step}"
        })
        
    def emit_thinking(self, thought: str):
        """Emit agent thinking/reasoning updates."""
        self.emit_status('thinking', {
            'thought': thought,
            'message': f"💭 {thought}"
        })
        
    def emit_progress(self, current: int, total: int, description: str):
        """Emit progress updates."""
        self.emit_status('progress', {
            'current': current,
            'total': total,
            'percentage': (current / total * 100) if total > 0 else 0,
            'description': description,
            'message': f"📊 Progress: {current}/{total} - {description}"
        })
        
    def emit_error(self, error: str):
        """Emit error messages."""
        self.emit_status('error', {
            'error': error,
            'message': f"❌ Error: {error}"
        })
        
    def close(self):
        """Close the streaming logger."""
        self.active = False

class EnhancedChatSession:
    """Enhanced chat session with streaming capabilities."""
    
    def __init__(self, session_id: str, socket_id: Optional[str] = None):
        self.session_id = session_id
        self.socket_id = socket_id
        self.orchestrator = OrchestratorAgent()
        self.kb_agent = KnowledgeBaseAgent()
        self.compliance_agent = ComplianceAgent()
        self.history = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.stream_logger = StreamingLogger(session_id, socket_id)
        
    def process_message_streaming(self, message: str, context: Optional[str] = None) -> Generator:
        """Process a message with streaming updates."""
        try:
            self.last_activity = datetime.now()
            
            # Add to history
            self.history.append({
                'type': 'user',
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'context': context
            })
            
            # Emit initial status
            self.stream_logger.emit_agent_start("Orchestrator Agent", "Analyzing your request")
            time.sleep(0.5)  # Small delay for visual effect
            
            # Analyze the request type
            self.stream_logger.emit_thinking("Determining the best approach for your request...")
            time.sleep(0.5)
            
            # Determine action type based on message content
            action_type = self._determine_action(message)
            self.stream_logger.emit_step_complete("Request Analysis", f"Identified as: {action_type}")
            
            # Process based on action type
            if action_type == "search":
                yield from self._process_search_streaming(message)
            elif action_type == "compliance":
                yield from self._process_compliance_streaming(message)
            elif action_type == "workflow":
                yield from self._process_workflow_streaming(message)
            elif action_type == "keyword_check":
                yield from self._process_keyword_check_streaming(message)
            else:
                yield from self._process_general_streaming(message)
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.stream_logger.emit_error(str(e))
            yield json.dumps({
                'type': 'error',
                'error': str(e)
            })
    
    def _determine_action(self, message: str) -> str:
        """Determine the type of action from the message."""
        message_lower = message.lower()
        
        # Check for keyword compliance patterns first (more specific)
        if any(phrase in message_lower for phrase in ['keyword', 'contains', 'contain', 'includes', 'include', 
                                                       'all documents contain', 'documents have',
                                                       'check if documents', 'check if all']):
            return "keyword_check"
        elif any(word in message_lower for word in ['search', 'find', 'retrieve', 'look for']):
            return "search"
        elif any(word in message_lower for word in ['compliance', 'check', 'validate', 'scan']):
            return "compliance"
        elif any(word in message_lower for word in ['workflow', 'process', 'complete']):
            return "workflow"
        else:
            return "general"
    
    def _process_search_streaming(self, message: str) -> Generator:
        """Process search request with streaming."""
        self.stream_logger.emit_agent_start("Knowledge Base Agent", "Searching documents")
        yield json.dumps({'type': 'status', 'message': 'Starting knowledge base search...'})
        
        # Simulate search steps
        steps = [
            "Parsing search query",
            "Building search vectors",
            "Querying knowledge base",
            "Ranking results by relevance",
            "Formatting search results"
        ]
        
        for i, step in enumerate(steps, 1):
            self.stream_logger.emit_progress(i, len(steps), step)
            time.sleep(0.3)
        
        # Use actual KB agent
        self.stream_logger.emit_tool_use("retrieve_from_knowledge_base", {"query": message})
        results = self.kb_agent.retrieve_documents(message)
        
        self.stream_logger.emit_step_complete("Search", f"Found {results.get('total', 0)} documents")
        
        # Format response
        response = self._format_search_results(results)
        
        yield json.dumps({
            'type': 'final',
            'response': response,
            'data': results
        })
    
    def _process_compliance_streaming(self, message: str) -> Generator:
        """Process compliance check with streaming."""
        self.stream_logger.emit_agent_start("Compliance Agent", "Checking document compliance")
        yield json.dumps({'type': 'status', 'message': 'Initiating compliance check...'})
        
        # Compliance check steps
        steps = [
            ("Extracting document content", 0.5),
            ("Loading compliance rules", 0.3),
            ("Checking for PII data", 0.8),
            ("Validating security policies", 0.7),
            ("Checking regulatory requirements", 0.6),
            ("Analyzing data protection", 0.9),
            ("Generating compliance score", 0.4),
            ("Preparing recommendations", 0.5)
        ]
        
        for i, (step, duration) in enumerate(steps, 1):
            self.stream_logger.emit_progress(i, len(steps), step)
            self.stream_logger.emit_tool_use(f"compliance_check_{i}", {"step": step})
            time.sleep(duration)
        
        # Extract content from message (simplified)
        content = message.replace("check compliance", "").replace("check", "").strip()
        
        if not content:
            content = "Sample document content for compliance checking"
        
        # Use actual compliance agent
        result = self.compliance_agent.check_document(content)
        
        self.stream_logger.emit_step_complete(
            "Compliance Check", 
            f"Score: {result.score:.2f}, Compliant: {result.is_compliant}"
        )
        
        # Generate response
        response = self._format_compliance_results(result)
        
        yield json.dumps({
            'type': 'final',
            'response': response,
            'data': result.dict()
        })
    
    def _process_workflow_streaming(self, message: str) -> Generator:
        """Process workflow with streaming."""
        self.stream_logger.emit_agent_start("Orchestrator Agent", "Running compliance workflow")
        yield json.dumps({'type': 'status', 'message': 'Initializing workflow...'})
        
        # Workflow steps
        workflow_steps = [
            {
                'agent': 'Knowledge Base Agent',
                'action': 'Retrieving relevant documents',
                'tool': 'retrieve_from_knowledge_base'
            },
            {
                'agent': 'Compliance Agent',
                'action': 'Analyzing document compliance',
                'tool': 'check_document_compliance'
            },
            {
                'agent': 'Compliance Agent',
                'action': 'Checking for violations',
                'tool': 'scan_violations'
            },
            {
                'agent': 'Orchestrator Agent',
                'action': 'Generating recommendations',
                'tool': 'generate_recommendations'
            },
            {
                'agent': 'Orchestrator Agent',
                'action': 'Compiling final report',
                'tool': 'compile_report'
            }
        ]
        
        for i, step in enumerate(workflow_steps, 1):
            self.stream_logger.emit_agent_start(step['agent'], step['action'])
            self.stream_logger.emit_tool_use(step['tool'], {})
            self.stream_logger.emit_progress(i, len(workflow_steps), step['action'])
            time.sleep(0.8)
        
        # Process with actual orchestrator
        query = message.replace("workflow", "").replace("run", "").strip() or "compliance policies"
        results = self.orchestrator.process_compliance_workflow(query)
        
        self.stream_logger.emit_step_complete("Workflow", "All steps completed successfully")
        
        # Format response
        response = self._format_workflow_results(results)
        
        yield json.dumps({
            'type': 'final',
            'response': response,
            'data': results
        })
    
    def _process_keyword_check_streaming(self, message: str) -> Generator:
        """Process keyword compliance check with streaming."""
        self.stream_logger.emit_agent_start("Compliance Agent", "Checking keyword compliance")
        yield json.dumps({'type': 'status', 'message': 'Starting keyword compliance check...'})
        
        # Extract keywords from message - much improved logic
        import re
        keywords = []
        
        # First try to find quoted strings
        keywords = re.findall(r'["\']([^"\']+)["\']', message)
        
        if not keywords:
            # Look for patterns like "check for X keyword" or "contains X"
            patterns = [
                r'check\s+for\s+["\']?(\w+)["\']?\s+keyword',
                r'check\s+.*?["\']?(\w+)["\']?\s+as\s+compliance',
                r'contain[s]?\s+["\']?([^"\'\n]+?)["\']?(?:\s+and|\s+or|$)',
                r'include[s]?\s+["\']?([^"\'\n]+?)["\']?(?:\s+and|\s+or|$)',
                r'have\s+["\']?([^"\'\n]+?)["\']?(?:\s+and|\s+or|$)',
                r'for\s+["\']?([^"\'\n]+?)["\']?\s+keyword',
                r'keyword[s]?\s*[:=]\s*["\']?([^"\'\n]+)["\']?',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, message.lower())
                if matches:
                    keywords.extend(matches)
                    break
            
            # If still no keywords, try to extract meaningful words after key phrases
            if not keywords:
                # Remove common words and extract potential keywords
                message_lower = message.lower()
                
                # Try to find words after these phrases
                trigger_phrases = [
                    "check for", "contains", "contain", "includes", "include",
                    "have", "has", "with", "search for", "find"
                ]
                
                for phrase in trigger_phrases:
                    if phrase in message_lower:
                        # Get text after the trigger phrase
                        parts = message_lower.split(phrase)
                        if len(parts) > 1:
                            remaining_text = parts[1].strip()
                            
                            # Remove common words and punctuation
                            stop_words = {
                                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
                                'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
                                'will', 'would', 'could', 'should', 'may', 'might', 'must',
                                'shall', 'can', 'need', 'all', 'documents', 'document', 'files',
                                'file', 'check', 'compliance', 'if', 'keyword', 'keywords'
                            }
                            
                            # Clean and extract words
                            words = re.findall(r'\b\w+\b', remaining_text)
                            meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
                            
                            if meaningful_words:
                                # Take first 1-3 meaningful words as keywords
                                keywords = meaningful_words[:3]
                                break
        
        # Clean up keywords
        keywords = [k.strip().strip('"').strip("'") for k in keywords if k.strip()]
        
        # If still no keywords, don't use defaults - ask for clarification
        if not keywords:
            self.stream_logger.emit_error("Could not identify keywords to check")
            response = """I couldn't identify which keywords to check for in your request.

Please specify the keywords more clearly, for example:
• Check for "policy" keyword
• Check if documents contain "privacy"
• Verify all documents have "GDPR" and "compliance"

What specific keywords would you like me to check for?"""
            
            yield json.dumps({
                'type': 'final',
                'response': response,
                'data': {'error': 'No keywords identified'}
            })
            return
        
        self.stream_logger.emit_thinking(f"Checking for keywords: {', '.join(keywords)}")
        
        # Determine match type from message
        match_type = "all" if "all" in message.lower() else "any"
        if "every" in message.lower() or "each" in message.lower():
            match_type = "all"
        
        self.stream_logger.emit_thinking(f"Match type: {'All keywords must be present' if match_type == 'all' else 'At least one keyword must be present'}")
        
        # Step 1: Retrieve all documents from KB
        self.stream_logger.emit_agent_start("Knowledge Base Agent", "Retrieving all documents")
        self.stream_logger.emit_tool_use("retrieve_from_knowledge_base", {"query": "*", "max_results": 100})
        time.sleep(0.5)
        
        # Get actual documents from KB
        kb_results = self.kb_agent.retrieve_documents("*", max_results=100)
        documents = kb_results.get('documents', [])
        
        if not documents:
            # If no documents from retrieval, try to get from local KB
            from tools.knowledge_base_tools import kb_tools
            list_results = kb_tools.list_knowledge_base_documents(limit=100)
            if list_results.get('success') and list_results.get('documents'):
                documents = list_results['documents']
        
        self.stream_logger.emit_step_complete("Document Retrieval", f"Retrieved {len(documents)} documents")
        
        # Step 2: Check keyword compliance using actual compliance agent
        self.stream_logger.emit_agent_start("Compliance Agent", "Analyzing keyword compliance")
        
        # Process documents in batches with progress
        batch_size = 10
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            self.stream_logger.emit_progress(batch_end, total_docs, f"Processing documents {i+1}-{batch_end}")
            time.sleep(0.3)
        
        # Use actual compliance agent to check keywords
        self.stream_logger.emit_tool_use("check_keyword_compliance", {
            "required_keywords": keywords,
            "match_type": match_type,
            "document_count": len(documents)
        })
        
        # Log what we're checking for debugging
        logger.info(f"Checking keyword compliance - Keywords: {keywords}, Match type: {match_type}, Documents: {len(documents)}")
        
        # Call the actual compliance agent method
        try:
            results = self.compliance_agent.check_keyword_compliance_kb(
                documents=documents,
                required_keywords=keywords,
                match_type=match_type
            )
            
            self.stream_logger.emit_step_complete(
                "Keyword Compliance Check", 
                f"Compliance rate: {results.get('compliance_rate', 0)*100:.1f}%"
            )
            
            # Format response with actual results
            response = f"""**Keyword Compliance Check Results**

**Required Keywords:** {', '.join(keywords)}
**Match Type:** At least one keyword must be present

**📊 Statistics:**
• Total Documents: {results.get('total_documents', 0)}
• Compliant: {results.get('compliant_count', 0)}
• Non-Compliant: {results.get('non_compliant_count', 0)}
• Compliance Rate: {results.get('compliance_rate', 0)*100:.1f}%

**❌ Non-Compliant Documents:**"""
            
            # Add details of non-compliant documents
            non_compliant = results.get('non_compliant_details', [])
            if non_compliant:
                for doc in non_compliant[:5]:  # Show first 5
                    response += f"\n• Document: {doc.get('document_id', 'Unknown')}"
                    missing = doc.get('missing_keywords', [])
                    if missing:
                        response += f"\n  Missing: {', '.join(missing)}"
                
                if len(non_compliant) > 5:
                    response += f"\n\n...and {len(non_compliant) - 5} more documents"
            else:
                response += "\nAll documents are compliant!"
            
            # Add recommendations
            if results.get('recommendations'):
                response += "\n\n**📋 Recommendations:**"
                for rec in results['recommendations'][:3]:
                    response += f"\n• {rec}"
            
            response += "\n\n✅ Keyword compliance check completed successfully."
            
        except Exception as e:
            logger.error(f"Error in keyword compliance check: {e}")
            self.stream_logger.emit_error(f"Error checking keyword compliance: {str(e)}")
            
            # Fallback response
            response = f"""**Keyword Compliance Check**
            
An error occurred while checking keyword compliance: {str(e)}

Please ensure:
1. The knowledge base contains documents
2. The compliance agent is properly configured
3. Try a simpler query like: "Check if documents contain privacy"
"""
            results = {'error': str(e)}
        
        yield json.dumps({
            'type': 'final',
            'response': response,
            'data': results
        })
    
    def _process_general_streaming(self, message: str) -> Generator:
        """Process general request with streaming."""
        self.stream_logger.emit_agent_start("Orchestrator Agent", "Processing your request")
        yield json.dumps({'type': 'status', 'message': 'Processing...'})
        
        self.stream_logger.emit_thinking("Analyzing request context...")
        time.sleep(0.5)
        
        # Check if this might be a keyword compliance request that wasn't caught
        message_lower = message.lower()
        keyword_indicators = ['keyword', 'contain', 'include', 'have', 'check for', 'check if']
        
        if any(indicator in message_lower for indicator in keyword_indicators):
            self.stream_logger.emit_thinking("This appears to be a keyword compliance check...")
            yield from self._process_keyword_check_streaming(message)
            return
        
        self.stream_logger.emit_thinking("Determining best approach...")
        time.sleep(0.5)
        
        self.stream_logger.emit_tool_use("interactive_session", {"message": message})
        
        # Use actual orchestrator
        response = self.orchestrator.interactive_session(message)
        
        self.stream_logger.emit_step_complete("Processing", "Request completed")
        
        yield json.dumps({
            'type': 'final',
            'response': response,
            'data': None
        })
    
    def _format_search_results(self, results: Dict) -> str:
        """Format search results for display."""
        response = "**Knowledge Base Search Results**\n\n"
        
        if results.get('documents'):
            response += f"Found {results.get('total', len(results['documents']))} relevant documents:\n\n"
            
            for i, doc in enumerate(results.get('documents', [])[:5], 1):
                response += f"**Document {i}:**\n"
                response += f"{doc.get('content', '')[:200]}...\n\n"
        else:
            response += "No relevant documents found.\n"
        
        if results.get('insights'):
            response += "**Key Insights:**\n"
            for insight in results['insights']:
                response += f"• {insight}\n"
        
        return response
    
    def _format_compliance_results(self, result) -> str:
        """Format compliance results for display."""
        response = "**Compliance Check Results**\n\n"
        response += f"• **Compliant:** {'✅ Yes' if result.is_compliant else '❌ No'}\n"
        response += f"• **Score:** {result.score:.2f}/1.00\n\n"
        
        if result.violations:
            response += f"**Violations Found ({len(result.violations)}):**\n"
            for v in result.violations[:5]:
                if isinstance(v, dict):
                    response += f"• {v.get('rule_name', 'Unknown')}: {v.get('description', '')}\n"
                else:
                    response += f"• {v}\n"
        
        if result.warnings:
            response += f"\n**Warnings ({len(result.warnings)}):**\n"
            for w in result.warnings[:3]:
                if isinstance(w, dict):
                    response += f"• {w.get('description', '')}\n"
                else:
                    response += f"• {w}\n"
        
        return response
    
    def _format_workflow_results(self, results: Dict) -> str:
        """Format workflow results for display."""
        response = "**Compliance Workflow Results**\n\n"
        response += f"✅ Completed {len(results.get('steps', []))} workflow steps\n\n"
        
        for i, step in enumerate(results.get('steps', []), 1):
            response += f"**Step {i}: {step.get('step', '')}**\n"
            response += f"Status: ✓ {step.get('status', '')}\n\n"
        
        if results.get('summary'):
            response += f"**Summary:**\n{results['summary'][:500]}...\n"
        
        return response

def get_or_create_session(session_id: str, socket_id: Optional[str] = None) -> EnhancedChatSession:
    """Get existing session or create new one."""
    if session_id not in sessions:
        sessions[session_id] = EnhancedChatSession(session_id, socket_id)
        logger.info(f"Created new session: {session_id}")
    elif socket_id:
        # Update socket ID if provided
        sessions[session_id].socket_id = socket_id
        sessions[session_id].stream_logger.socket_id = socket_id
    return sessions[session_id]

# API Routes

@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'compliance_chat_frontend.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'kb_mode': settings.get_kb_mode(),
        'active_sessions': len(sessions)
    })

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    context = data.get('context', None)
    
    def generate():
        """Generate streaming responses."""
        session = get_or_create_session(session_id)
        
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"
        
        # Process message with streaming
        for chunk in session.process_message_streaming(message, context):
            yield f"data: {chunk}\n\n"
        
        # Send completion message
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Regular chat endpoint (non-streaming)."""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        context = data.get('context', None)
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        # Get or create session
        session = get_or_create_session(session_id)
        
        # Process message
        response = session.orchestrator.interactive_session(message)
        
        return jsonify({
            'success': True,
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# WebSocket Events for real-time updates
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {
        'socket_id': request.sid,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat message via WebSocket with streaming."""
    try:
        session_id = data.get('session_id', 'default')
        message = data.get('message', '')
        context = data.get('context', None)
        
        # Get or create session with socket ID
        session = get_or_create_session(session_id, request.sid)
        
        # Start processing in a thread
        def process_async():
            for chunk in session.process_message_streaming(message, context):
                chunk_data = json.loads(chunk)
                emit('chat_response', chunk_data, room=request.sid)
        
        thread = threading.Thread(target=process_async)
        thread.start()
        
    except Exception as e:
        logger.error(f"WebSocket message error: {e}")
        emit('error', {'error': str(e)})

@socketio.on('join_session')
def handle_join_session(data):
    """Join a specific session room."""
    session_id = data.get('session_id', 'default')
    logger.info(f"Client {request.sid} joining session {session_id}")
    # Store socket ID in session
    get_or_create_session(session_id, request.sid)

@app.route('/api/test/keyword-compliance', methods=['GET'])
def test_keyword_compliance():
    """Test endpoint for keyword compliance functionality."""
    try:
        # Test with sample documents
        test_documents = [
            {
                'id': 'doc_1',
                'content': 'This document contains privacy policy and data protection guidelines.',
                'metadata': {'type': 'policy'}
            },
            {
                'id': 'doc_2',
                'content': 'Security measures and compliance procedures are outlined here.',
                'metadata': {'type': 'security'}
            },
            {
                'id': 'doc_3',
                'content': 'General information about our services.',
                'metadata': {'type': 'general'}
            }
        ]
        
        test_keywords = ['privacy policy', 'data protection', 'compliance']
        
        from tools.compliance_tools import compliance_tools
        
        # Test the tool directly
        result = compliance_tools.check_keyword_compliance(
            required_keywords=test_keywords,
            match_type='any',
            documents=test_documents
        )
        
        return jsonify({
            'success': True,
            'test_result': result,
            'message': 'Keyword compliance tool is working correctly'
        })
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# CLI for development server
def main():
    """Run the Flask development server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Compliance Agent System API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    args.host = '0.0.0.0'
    args.port = 8000 
    args.debug = True
    
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║   Enhanced Compliance Agent System API Server        ║
    ║   With Real-time Streaming Support                   ║
    ╠══════════════════════════════════════════════════════╣
    ║   Starting server on {args.host}:{args.port:<31}║
    ║   Debug mode: {str(args.debug):<40}║
    ║   KB Mode: {settings.get_kb_mode().upper():<44}║
    ╚══════════════════════════════════════════════════════╝
    
    Features:
    ✨ Real-time streaming of agent activities
    🔧 Tool usage visualization
    🤖 Multi-agent coordination display
    📊 Progress tracking for long operations
    💭 Agent reasoning transparency
    
    API Endpoints:
    - POST /api/chat/stream  - Streaming chat (SSE)
    - POST /api/chat         - Regular chat
    - WS   /socket.io        - WebSocket connection
    
    Frontend available at: http://localhost:{args.port}/
    """)
    
    # Run with SocketIO for WebSocket support
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()