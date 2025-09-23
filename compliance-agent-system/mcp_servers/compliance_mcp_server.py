#!/usr/bin/env python3
"""MCP Server for compliance checking operations."""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("compliance-mcp-server")

# Compliance rules storage
COMPLIANCE_RULES = {
    "PII": {
        "patterns": [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"  # Credit card
        ],
        "severity": "high"
    },
    "SECURITY": {
        "keywords": ["password", "api_key", "secret", "token", "credential"],
        "severity": "high"
    },
    "REGULATORY": {
        "keywords": ["gdpr", "hipaa", "sox", "pci-dss", "ccpa"],
        "severity": "medium"
    }
}

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available compliance tools."""
    return [
        types.Tool(
            name="check_compliance",
            description="Check text for compliance violations including PII, security risks, and regulatory requirements",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to check for compliance"
                    },
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific rules to check (PII, SECURITY, REGULATORY)",
                        "default": ["PII", "SECURITY", "REGULATORY"]
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Compliance score threshold (0-1)",
                        "default": 0.8
                    }
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="scan_pii",
            description="Scan content specifically for Personally Identifiable Information",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to scan for PII"
                    },
                    "mask": {
                        "type": "boolean",
                        "description": "Whether to mask found PII in the response",
                        "default": False
                    }
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="validate_gdpr",
            description="Validate content against GDPR requirements",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to validate"
                    },
                    "check_consent": {
                        "type": "boolean",
                        "description": "Check for consent language",
                        "default": True
                    },
                    "check_rights": {
                        "type": "boolean",
                        "description": "Check for data subject rights mentions",
                        "default": True
                    }
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="generate_compliance_report",
            description="Generate a detailed compliance report",
            inputSchema={
                "type": "object",
                "properties": {
                    "violations": {
                        "type": "array",
                        "description": "List of violations found"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown", "html"],
                        "description": "Report format",
                        "default": "markdown"
                    }
                },
                "required": ["violations"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, 
    arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution."""
    
    if name == "check_compliance":
        return await check_compliance(arguments)
    elif name == "scan_pii":
        return await scan_pii(arguments)
    elif name == "validate_gdpr":
        return await validate_gdpr(arguments)
    elif name == "generate_compliance_report":
        return await generate_compliance_report(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def check_compliance(args: dict) -> list[types.TextContent]:
    """Check content for compliance violations."""
    content = args.get("content", "")
    rules_to_check = args.get("rules", ["PII", "SECURITY", "REGULATORY"])
    threshold = args.get("threshold", 0.8)
    
    violations = []
    score = 1.0
    
    for rule_name in rules_to_check:
        if rule_name not in COMPLIANCE_RULES:
            continue
            
        rule = COMPLIANCE_RULES[rule_name]
        
        # Check patterns
        if "patterns" in rule:
            for pattern in rule["patterns"]:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append({
                        "rule": rule_name,
                        "type": "pattern",
                        "severity": rule["severity"],
                        "matches": len(matches),
                        "details": f"Found {len(matches)} instances"
                    })
                    score -= 0.2 if rule["severity"] == "high" else 0.1
        
        # Check keywords
        if "keywords" in rule:
            for keyword in rule["keywords"]:
                if keyword.lower() in content.lower():
                    violations.append({
                        "rule": rule_name,
                        "type": "keyword",
                        "severity": rule["severity"],
                        "keyword": keyword,
                        "details": f"Found keyword: {keyword}"
                    })
                    score -= 0.15 if rule["severity"] == "high" else 0.05
    
    score = max(0, score)
    is_compliant = score >= threshold
    
    result = {
        "is_compliant": is_compliant,
        "score": round(score, 2),
        "threshold": threshold,
        "violations": violations,
        "violations_count": len(violations),
        "checked_at": datetime.now().isoformat()
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def scan_pii(args: dict) -> list[types.TextContent]:
    """Scan content for PII."""
    content = args.get("content", "")
    mask = args.get("mask", False)
    
    pii_found = []
    masked_content = content
    
    # Check for various PII patterns
    pii_patterns = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "Phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "IP Address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "Date of Birth": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    }
    
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            pii_found.append({
                "type": pii_type,
                "count": len(matches),
                "examples": matches[:3] if not mask else ["[REDACTED]"] * min(3, len(matches))
            })
            
            if mask:
                masked_content = re.sub(pattern, f"[{pii_type}_REDACTED]", masked_content, flags=re.IGNORECASE)
    
    result = {
        "pii_detected": len(pii_found) > 0,
        "pii_types_found": len(pii_found),
        "details": pii_found,
        "masked_content": masked_content if mask else None,
        "scan_timestamp": datetime.now().isoformat()
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def validate_gdpr(args: dict) -> list[types.TextContent]:
    """Validate content against GDPR requirements."""
    content = args.get("content", "")
    check_consent = args.get("check_consent", True)
    check_rights = args.get("check_rights", True)
    
    gdpr_checks = {
        "has_gdpr_mention": False,
        "has_consent_language": False,
        "has_data_rights": False,
        "has_privacy_policy": False,
        "has_data_controller": False,
        "issues": [],
        "recommendations": []
    }
    
    # Check for GDPR mentions
    if re.search(r"\bgdpr\b", content, re.IGNORECASE):
        gdpr_checks["has_gdpr_mention"] = True
    
    # Check for consent language
    if check_consent:
        consent_keywords = ["consent", "agree", "opt-in", "permission", "authorize"]
        if any(keyword in content.lower() for keyword in consent_keywords):
            gdpr_checks["has_consent_language"] = True
        else:
            gdpr_checks["issues"].append("Missing explicit consent language")
            gdpr_checks["recommendations"].append("Add clear consent mechanisms")
    
    # Check for data rights
    if check_rights:
        rights_keywords = ["right to access", "right to erasure", "right to object", "data portability", "rectification"]
        if any(keyword in content.lower() for keyword in rights_keywords):
            gdpr_checks["has_data_rights"] = True
        else:
            gdpr_checks["issues"].append("Missing data subject rights information")
            gdpr_checks["recommendations"].append("Include information about user data rights")
    
    # Check for privacy policy
    if "privacy policy" in content.lower():
        gdpr_checks["has_privacy_policy"] = True
    else:
        gdpr_checks["recommendations"].append("Reference or include privacy policy")
    
    # Check for data controller info
    if "data controller" in content.lower() or "contact" in content.lower():
        gdpr_checks["has_data_controller"] = True
    else:
        gdpr_checks["recommendations"].append("Include data controller contact information")
    
    # Calculate compliance score
    checks_passed = sum([
        gdpr_checks["has_gdpr_mention"],
        gdpr_checks["has_consent_language"],
        gdpr_checks["has_data_rights"],
        gdpr_checks["has_privacy_policy"],
        gdpr_checks["has_data_controller"]
    ])
    
    gdpr_checks["compliance_score"] = checks_passed / 5
    gdpr_checks["is_compliant"] = checks_passed >= 3
    
    return [types.TextContent(
        type="text",
        text=json.dumps(gdpr_checks, indent=2)
    )]

async def generate_compliance_report(args: dict) -> list[types.TextContent]:
    """Generate a compliance report."""
    violations = args.get("violations", [])
    format_type = args.get("format", "markdown")
    
    if format_type == "markdown":
        report = "# Compliance Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if not violations:
            report += "✅ **No violations found**\n\n"
            report += "The content appears to be compliant with all checked rules.\n"
        else:
            report += f"⚠️ **{len(violations)} violations found**\n\n"
            
            # Group by severity
            high_severity = [v for v in violations if v.get("severity") == "high"]
            medium_severity = [v for v in violations if v.get("severity") == "medium"]
            low_severity = [v for v in violations if v.get("severity") == "low"]
            
            if high_severity:
                report += "## High Severity Issues\n\n"
                for v in high_severity:
                    report += f"- **{v.get('rule')}**: {v.get('details', 'N/A')}\n"
                report += "\n"
            
            if medium_severity:
                report += "## Medium Severity Issues\n\n"
                for v in medium_severity:
                    report += f"- **{v.get('rule')}**: {v.get('details', 'N/A')}\n"
                report += "\n"
            
            if low_severity:
                report += "## Low Severity Issues\n\n"
                for v in low_severity:
                    report += f"- **{v.get('rule')}**: {v.get('details', 'N/A')}\n"
                report += "\n"
            
            report += "## Recommendations\n\n"
            report += "1. Address all high severity issues immediately\n"
            report += "2. Review and remediate medium severity issues\n"
            report += "3. Consider implementing additional compliance checks\n"
            report += "4. Regular compliance audits are recommended\n"
        
        return [types.TextContent(type="text", text=report)]
    
    elif format_type == "json":
        report = {
            "report_type": "compliance",
            "generated_at": datetime.now().isoformat(),
            "violations": violations,
            "summary": {
                "total_violations": len(violations),
                "high_severity": len([v for v in violations if v.get("severity") == "high"]),
                "medium_severity": len([v for v in violations if v.get("severity") == "medium"]),
                "low_severity": len([v for v in violations if v.get("severity") == "low"])
            }
        }
        return [types.TextContent(type="text", text=json.dumps(report, indent=2))]
    
    else:  # HTML format
        html = f"""
        <html>
        <head><title>Compliance Report</title></head>
        <body>
            <h1>Compliance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Violations: {len(violations)}</p>
            <ul>
        """
        for v in violations:
            html += f"<li><b>{v.get('rule')}</b>: {v.get('details', 'N/A')}</li>"
        html += """
            </ul>
        </body>
        </html>
        """
        return [types.TextContent(type="text", text=html)]

async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="compliance-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())