"""Notification and alerting tools."""

from strands import tool
import logging

logger = logging.getLogger(__name__)

class NotificationTools:
    """Tools for notifications and alerts."""
    
    @tool
    def send_alert(self, message: str, severity: str = "info"):
        """Send an alert notification."""
        logger.info(f"Alert [{severity}]: {message}")
        return {"status": "sent", "message": message}

notification_tools = NotificationTools()
