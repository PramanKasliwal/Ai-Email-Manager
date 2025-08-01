# AI-Email Manager
🤖 Smart email manager with AI classification, auto-labeling & priority alerts. Uses transformers + Gmail API to categorize emails, flag important messages & notify about sales leads. Boost productivity with intelligent email automation!
---
## Prerequisites

Install required Python packages:

```bash
pip install torch transformers google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

## Gmail API Setup

1. **Enable Gmail API:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable the Gmail API
   - Create credentials (OAuth 2.0 Client ID)
   - Download the credentials file as `credentials.json`

2. **Set up OAuth consent screen:**
   - Add your email to test users
   - Set scopes to include Gmail API

## Configuration

Create a `config.json` file with your settings:

```json
{
  "gmail": {
    "credentials_file": "credentials.json"
  },
  "alerts": {
    "enabled": true,
    "recipient": "your-alert-email@example.com",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-sender-email@gmail.com",
    "sender_password": "your-app-password"
  },
  "processing": {
    "max_emails": 20,
    "confidence_threshold": 0.6
  }
}
```

### Gmail App Password Setup

For alerts to work, you need to create an App Password:

1. Go to your Google Account settings
2. Enable 2-Factor Authentication
3. Generate an App Password for "Mail"
4. Use this password in the config file

## Usage Examples

### Basic Usage (Process Once)
```bash
python email_ai_manager.py
```

### Continuous Processing
```bash
# Check every 5 minutes
python email_ai_manager.py --continuous 5

# Check every 10 minutes
python email_ai_manager.py --continuous 10
```

## Features

### 🤖 AI Classification
- **Sales**: Proposals, deals, contracts, quotes
- **Support**: Issues, problems, technical requests
- **Finance**: Invoices, payments, billing
- **Junk**: Spam and low-priority emails

### 🏷️ Auto-Labeling
- Creates Gmail labels: `AI/Job_opportunities`,`AI/Sales`, `AI/Support`, `AI/Finance`, `AI/Junk`
- Automatically applies appropriate labels to classified emails

### ⚡ Priority Detection
- Identifies urgent emails using keyword analysis
- Flags important emails in Gmail
- Sends instant alerts for priority sales leads

### 📧 Smart Alerts
- Email notifications for high-priority items
- Customizable alert thresholds
- Detailed email previews in alerts

## Customization

### Adding Custom Keywords

Modify the `priority_keywords` in the `EmailClassifier` class:

```python
self.priority_keywords = {
    "sales": ["urgent", "deadline", "proposal", "contract", "deal", "meeting", "demo", "quote", "revenue"],
    "support": ["critical", "down", "error", "urgent", "emergency", "issue", "problem", "outage"],
    "finance": ["invoice", "payment", "urgent", "overdue", "accounting", "billing", "budget"],
    "junk": []
}
```

### Custom Classification Categories

Update the `labels` list in `EmailClassifier`:

```python
self.labels = ["sales", "support", "finance", "junk", "hr", "marketing"]
```

### Confidence Threshold

Adjust in `config.json`:
```json
{
  "processing": {
    "confidence_threshold": 0.7  // Higher = more strict classification
  }
}
```

## Advanced Features

### Scheduling with Cron

Add to your crontab for automated processing:

```bash
# Run every 15 minutes
*/15 * * * * /usr/bin/python3 /path/to/email_ai_manager.py

# Run every hour
0 * * * * /usr/bin/python3 /path/to/email_ai_manager.py
```

### Integration with Slack/Teams

Extend the `AlertManager` class to send alerts to messaging platforms:

```python
import requests

def send_slack_alert(self, webhook_url: str, message: str):
    payload = {"text": message}
    requests.post(webhook_url, json=payload)
```

### Database Logging

Add email processing history tracking:

```python
import sqlite3

def log_classification(self, email_id: str, category: str, confidence: float):
    # Add database logging logic here
    pass
```

## Troubleshooting

### Common Issues

1. **Authentication Errors:**
   - Ensure `credentials.json` is in the correct location
   - Check OAuth consent screen configuration
   - Verify Gmail API is enabled

2. **Classification Accuracy:**
   - Increase `confidence_threshold` for stricter classification
   - Add custom keywords for your specific use case
   - Consider fine-tuning the model on your email data

3. **Alert Delivery:**
   - Verify SMTP settings and app password
   - Check spam folders for alert emails
   - Test with a simple email send first

4. **Performance:**
   - Reduce `max_emails` for faster processing
   - Use GPU acceleration if available
   - Consider running on a server for continuous operation

## Security Considerations

- Keep `credentials.json` and `token.pickle` secure
- Use environment variables for sensitive config
- Regularly rotate app passwords
- Monitor API usage quotas

## Monitoring and Logs

The script provides detailed logging. To save logs to file:

```bash
python email_ai_manager.py --continuous 5 > email_manager.log 2>&1
```

## Performance Optimization

For high-volume email processing:

1. **Batch Processing**: Process emails in batches
2. **Caching**: Cache model predictions for similar emails
3. **Async Processing**: Use asyncio for concurrent API calls
4. **Model Optimization**: Use smaller, faster models for real-time processing
