import os
import pickle
import base64
import json
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailClassifier:
    """AI-powered email classifier using transformers"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize the email classifier with a pre-trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize classification pipeline with zero-shot classification
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Define classification labels
        self.labels = ["sales", "support", "finance", "junk", "job_opportunity"]
        
        # Priority keywords for each category
        self.priority_keywords = {
            "job_opportunity": [# Direct job terms
                                "job", "position", "role", "opportunity", "opening", "vacancy",
                                "hiring", "recruiting", "career", "employment", "application"
                                
                                # Action words
                                "apply", "resume", "cv", "interview", "candidate",
                                "join our team", "work with us", "interested in you",
                                
                                # Specific roles (customize for your field)
                                "developer", "engineer", "manager", "analyst", "designer",
                                "software", "python", "javascript", "react", "aws",
                                
                                # Company indicators
                                "startup", "company", "remote", "full-time", "contract", "flexible", "hybrid",
                                
                                # Compensation
                                "salary", "compensation", "benefits", "equity", "stock",
                                
                                # Urgency
                                "urgent hiring", "immediate start", "quick process"],
            "sales": ["urgent", "deadline", "proposal", "contract", "deal", "meeting", "demo", "quote"],
            "support": ["critical", "down", "error", "urgent", "emergency", "issue", "problem"],
            "finance": ["invoice", "payment", "urgent", "overdue", "accounting", "billing"],
            "junk": []  # Junk emails are typically low priority
        }
    
    def classify_email(self, subject: str, body: str) -> Tuple[str, float, bool]:
        """
        Classify email content into categories
        
        Returns:
            Tuple of (category, confidence_score, is_priority)
        """
        # Combine subject and body for classification
        text = f"{subject} {body}".lower()
        
        # Perform zero-shot classification
        result = self.classifier(text, self.labels)
        
        category = result['labels'][0]
        confidence = result['scores'][0]
        
        # Determine if email is priority based on keywords
        is_priority = self._is_priority_email(text, category)
        
        logger.info(f"Classified as: {category} (confidence: {confidence:.2f}, priority: {is_priority})")
        
        return category, confidence, is_priority
    
    def _is_priority_email(self, text: str, category: str) -> bool:
        """Check if email contains priority keywords for its category"""
        if category == "junk":
            return False
        
        keywords = self.priority_keywords.get(category, [])
        return any(keyword in text for keyword in keywords)

class GmailManager:
    """Gmail API manager for email operations"""
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    
    def __init__(self, credentials_file: str = 'credentials.json'):
        """Initialize Gmail API connection"""
        self.credentials_file = credentials_file
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        
        # Load existing token
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for future use
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API authentication successful")
    
    def get_unread_emails(self, max_results: int = 10) -> List[Dict]:
        """Fetch unread emails from inbox"""
        try:
            # Get unread messages
            results = self.service.users().messages().list(
                userId='me',
                q='is:unread in:inbox',
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                # Get full message details
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()
                
                email_data = self._parse_email(msg)
                emails.append(email_data)
            
            logger.info(f"Retrieved {len(emails)} unread emails")
            return emails
            
        except HttpError as error:
            logger.error(f"Gmail API error: {error}")
            return []
    
    def _parse_email(self, message: Dict) -> Dict:
        """Parse Gmail message into structured data"""
        headers = {h['name']: h['value'] for h in message['payload']['headers']}
        
        # Extract email body
        body = self._extract_body(message['payload'])
        
        return {
            'id': message['id'],
            'subject': headers.get('Subject', ''),
            'sender': headers.get('From', ''),
            'date': headers.get('Date', ''),
            'body': body,
            'thread_id': message['threadId']
        }
    
    def _extract_body(self, payload: Dict) -> str:
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                    break
        elif payload['mimeType'] == 'text/plain':
            data = payload['body']['data']
            body = base64.urlsafe_b64decode(data).decode('utf-8')
        
        return body
    
    def create_label(self, label_name: str) -> str:
        """Create a Gmail label if it doesn't exist"""
        try:
            # Check if label exists
            labels = self.service.users().labels().list(userId='me').execute()
            for label in labels['labels']:
                if label['name'] == label_name:
                    return label['id']
            
            # Create new label
            label_object = {
                'name': label_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            
            created_label = self.service.users().labels().create(
                userId='me',
                body=label_object
            ).execute()
            
            logger.info(f"Created label: {label_name}")
            return created_label['id']
            
        except HttpError as error:
            logger.error(f"Error creating label: {error}")
            return None
    
    def apply_label(self, message_id: str, label_id: str):
        """Apply label to email"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            
            logger.info(f"Applied label to message {message_id}")
            
        except HttpError as error:
            logger.error(f"Error applying label: {error}")
    
    def flag_important(self, message_id: str):
        """Mark email as important"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': ['IMPORTANT']}
            ).execute()
            
            logger.info(f"Flagged message {message_id} as important")
            
        except HttpError as error:
            logger.error(f"Error flagging important: {error}")

class AlertManager:
    """Handles email alerts for priority messages"""
    
    def __init__(self, smtp_server: str, smtp_port: int, email: str, password: str):
        """Initialize alert manager with SMTP settings"""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
    
    def send_alert(self, recipient: str, subject: str, body: str):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = recipient
            msg['Subject'] = f"[PRIORITY ALERT] {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert sent to {recipient}")
            
        except Exception as error:
            logger.error(f"Error sending alert: {error}")

class EmailAIManager:
    """Main orchestrator for AI-powered email management"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the email AI manager"""
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.classifier = EmailClassifier()
        self.gmail = GmailManager(self.config['gmail']['credentials_file'])
        
        if self.config.get('alerts', {}).get('enabled', False):
            alert_config = self.config['alerts']
            self.alert_manager = AlertManager(
                alert_config['smtp_server'],
                alert_config['smtp_port'],
                alert_config['sender_email'],
                alert_config['sender_password']
            )
        else:
            self.alert_manager = None
        
        # Create labels for categories
        self.label_ids = {}
        for category in self.classifier.labels:
            label_name = f"AI/{category.capitalize()}"
            self.label_ids[category] = self.gmail.create_label(label_name)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "gmail": {
                "credentials_file": "credentials.json"
            },
            "alerts": {
                "enabled": False,
                "recipient": "",
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": ""
            },
            "processing": {
                "max_emails": 20,
                "confidence_threshold": 0.5
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file: {config_file}")
        
        return default_config
    
    def process_emails(self):
        """Main processing loop for emails"""
        logger.info("Starting email processing...")
        
        # Get unread emails
        emails = self.gmail.get_unread_emails(
            max_results=self.config['processing']['max_emails']
        )
        
        if not emails:
            logger.info("No unread emails found")
            return
        
        processed_count = 0
        priority_count = 0
        
        for email in emails:
            try:
                # Classify email
                category, confidence, is_priority = self.classifier.classify_email(
                    email['subject'], email['body']
                )
                
                # Skip if confidence is too low
                if confidence < self.config['processing']['confidence_threshold']:
                    logger.info(f"Skipping email due to low confidence: {confidence:.2f}")
                    continue
                
                # Apply label
                if category in self.label_ids and self.label_ids[category]:
                    self.gmail.apply_label(email['id'], self.label_ids[category])
                
                # Handle priority emails
                if is_priority:
                    self.gmail.flag_important(email['id'])
                    priority_count += 1
                    
                    # Send alert for priority sales emails
                    if category == "sales" and self.alert_manager:
                        alert_body = f"""
Priority sales email detected!

From: {email['sender']}
Subject: {email['subject']}
Category: {category.upper()}
Confidence: {confidence:.2f}

Email preview:
{email['body'][:200]}...

Please check your inbox immediately.
                        """
                        
                        self.alert_manager.send_alert(
                            self.config['alerts']['recipient'],
                            f"Priority Sales Lead - {email['subject'][:50]}",
                            alert_body
                        )
                
                processed_count += 1
                logger.info(f"Processed: {email['subject'][:50]}... -> {category.upper()}")
                
            except Exception as error:
                logger.error(f"Error processing email {email['id']}: {error}")
        
        logger.info(f"Processing complete: {processed_count} emails processed, {priority_count} priority emails found")
    
    def run_continuous(self, interval_minutes: int = 5):
        """Run email processing continuously"""
        logger.info(f"Starting continuous processing (every {interval_minutes} minutes)")
        
        while True:
            try:
                self.process_emails()
                
                # Wait for next iteration
                import time
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous processing...")
                break
            except Exception as error:
                logger.error(f"Error in continuous processing: {error}")
                # Wait before retrying
                import time
                time.sleep(60)

def main():
    """Main entry point"""
    # Initialize the email AI manager
    manager = EmailAIManager()
    
    # Run once or continuously
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        manager.run_continuous(interval)
    else:
        manager.process_emails()

if __name__ == "__main__":
    main()
